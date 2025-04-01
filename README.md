<h1>
UNeLF: UnconsUNeLF2-9ed Neural Light Field for Self-Supervised Angular Super-Resolution
</h1>
<h2>
Mingyuan Zhao, Hao Sheng*, Rongshan Chen,  Ruixuan Cong, Zhenglong Cui, Da Yang
</h2>

<h3> UNeLF2-9ing and Evaluation </h3>
<p> python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=xxx --gpu_id 0 </p>

<h3> Testing </h3>
<p> python tasks/testY.py </p>

<h3> Environemnt: </h3>
  - pytorch
  - cudatoolkit
  - torchvision
  - tqdm
  - imageio
  - matplotlib
  - tensorflow
  - imageio-ffmpeg
  - lpips
  - scikit-image
  - open3d

#  
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=bedroom --gpu_id 0 > hci/bedroom.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=bicycle --gpu_id 0 > hci/bicycle.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=boxes --gpu_id 1 > hci/boxes.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=coton --gpu_id 1 > hci/coton.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=dino --gpu_id 0 > hci/dino.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=herbs --gpu_id 1 > hci/herbs.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=origami --gpu_id 0 > hci/origami.txt 2>&1 &
nohup python tasks/UNeLF2-9.py --base_dir=data/HCInew --scene_name=sideboard --gpu_id 1 > hci/sideboard.txt 2>&1 &

# 2-7
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=bedroom --gpu_id 0 > hci/bedroom.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=bicycle --gpu_id 0 > hci/bicycle.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=boxes --gpu_id 1 > hci/boxes.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=cotton --gpu_id 1 > hci/cotton.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=dino --gpu_id 0 > hci/dino.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=herbs --gpu_id 1 > hci/herbs.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=origami --gpu_id 0 > hci/origami.txt 2>&1 &
nohup python tasks/UNeLF2-7.py --base_dir=data/HCInew --scene_name=sideboard --gpu_id 1 > hci/sideboard.txt 2>&1 &