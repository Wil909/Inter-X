# The implementation is based on https://github.com/eth-ait/aitviewer
import os
import numpy as np
from tqdm import tqdm
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

C.update_conf({'smplx_models':'./body_models'})

# Use software rendering instead of OpenGL
os.environ["PYOPENGL_PLATFORM"] = "egl"

class SMPLX_Viewer:
    def __init__(self):
        # Initialize headless renderer
        self.renderer = HeadlessRenderer(width=1920, height=1080)
        
        # Initialize scene
        self.scene = self.renderer.scene
        self.scene.fps = 120
        self.scene.playback_fps = 120
        
        # Reset and load initial sequence
        self.reset_for_interx()
        self.load_one_sequence()

    def reset_for_interx(self):
        self.text_val = ''
        self.clip_folder = './data/'
        self.text_folder = './texts'
        self.label_npy_list = []
        self.get_label_file_list()
        self.total_tasks = len(self.label_npy_list)
        self.label_pid = 0
        self.go_to_idx = 0

    def get_label_file_list(self):
        for clip in sorted(os.listdir(self.clip_folder)):
            if not clip.startswith('.'):
                self.label_npy_list.append(os.path.join(self.clip_folder, clip))
    
    def load_text_from_file(self):
        self.text_val = ''
        clip_name = self.label_npy_list[self.label_pid].split('/')[-1]
        if os.path.exists(os.path.join(self.text_folder, clip_name+'.txt')):
            with open(os.path.join(self.text_folder, clip_name+'.txt'), 'r') as f:
                for line in f.readlines():
                    self.text_val += line
                    self.text_val += '\n'

    def load_one_sequence(self):
        npy_folder = self.label_npy_list[self.label_pid]

        # load smplx
        smplx_path_p1 = os.path.join(npy_folder, 'P1.npz')
        smplx_path_p2 = os.path.join(npy_folder, 'P2.npz')
        params_p1 = np.load(smplx_path_p1, allow_pickle=True)
        params_p2 = np.load(smplx_path_p2, allow_pickle=True)
        nf_p1 = params_p1['pose_body'].shape[0]
        nf_p2 = params_p2['pose_body'].shape[0]
        
        self.total_frames = max(nf_p1, nf_p2)

        betas_p1 = params_p1['betas']
        poses_root_p1 = params_p1['root_orient']
        poses_body_p1 = params_p1['pose_body'].reshape(nf_p1,-1)
        poses_lhand_p1 = params_p1['pose_lhand'].reshape(nf_p1,-1)
        poses_rhand_p1 = params_p1['pose_rhand'].reshape(nf_p1,-1)
        transl_p1 = params_p1['trans']
        gender_p1 = str(params_p1['gender'])

        betas_p2 = params_p2['betas']
        poses_root_p2 = params_p2['root_orient']
        poses_body_p2 = params_p2['pose_body'].reshape(nf_p2,-1)
        poses_lhand_p2 = params_p2['pose_lhand'].reshape(nf_p2,-1)
        poses_rhand_p2 = params_p2['pose_rhand'].reshape(nf_p2,-1)
        transl_p2 = params_p2['trans']
        gender_p2 = str(params_p2['gender'])

        # create body models
        smplx_layer_p1 = SMPLLayer(model_type='smplx',gender=gender_p1,num_betas=10,device=C.device)
        smplx_layer_p2 = SMPLLayer(model_type='smplx',gender=gender_p2,num_betas=10,device=C.device)

        # create smplx sequence for two persons
        smplx_seq_p1 = SMPLSequence(poses_body=poses_body_p1,
                            smpl_layer=smplx_layer_p1,
                            poses_root=poses_root_p1,
                            betas=betas_p1,
                            trans=transl_p1,
                            poses_left_hand=poses_lhand_p1,
                            poses_right_hand=poses_rhand_p1,
                            device=C.device,
                            color=(0.11, 0.53, 0.8, 1.0)
                            )
        smplx_seq_p2 = SMPLSequence(poses_body=poses_body_p2,
                            smpl_layer=smplx_layer_p2,
                            poses_root=poses_root_p2,
                            betas=betas_p2,
                            trans=transl_p2,
                            poses_left_hand=poses_lhand_p2,
                            poses_right_hand=poses_rhand_p2,
                            device=C.device,
                            color=(1.0, 0.27, 0, 1.0)
                            )
        self.scene.add(smplx_seq_p1)
        self.scene.add(smplx_seq_p2)
        self.load_text_from_file()

        # Calculate midpoint between two people as view target
        p1_pos = smplx_seq_p1.positions[0,0] # Get first frame, first vertex position
        p2_pos = smplx_seq_p2.positions[0,0]
        midpoint = (p1_pos + p2_pos) / 2
        
        # Set camera to look at midpoint from a fixed offset
        offset = np.array([1, 2.0, 5.0])
        self.scene.camera.position = midpoint + offset
        self.scene.camera.target = midpoint

    def clear_one_sequence(self):
        for x in self.scene.nodes.copy():
            if type(x) is SMPLSequence or type(x) is SMPLLayer:
                self.scene.remove(x)

    def render_sequence(self):
        os.makedirs('output', exist_ok=True)
        self.renderer.save_video(video_dir=os.path.join('output', self.label_npy_list[self.label_pid].split('/')[-1]))

    def set_goto_record(self, idx):
        self.label_pid = int(idx) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def render_sequence_by_idx(self, idx):
        self.set_goto_record(idx)
        self.render_sequence()

    def render(self):
        # Render current frame
        self.renderer.render(0,0)
        
        # Save rendered frame as image
        image = self.renderer.get_current_frame_as_image()
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save image with sequence name
        clip_name = self.label_npy_list[self.label_pid].split('/')[-1]
        image.save(f'output/{clip_name}.png')

if __name__=='__main__':
    viewer = SMPLX_Viewer()
    
    for i in range(10):
        viewer.render_sequence_by_idx(i)
