# VideoMerge
Official Implementation of the VideoMerge

Thanks for open-sourced projects [Hunyuan-Video](https://github.com/Tencent/HunyuanVideo)


## Requirements

1. Clone this repository and navigate to its directory.
```
git clone https://github.com/szhan227/VideoMerge.git
cd VideoMerge
```

2. Setup environment
* Python >= 3.10
* Pytorch >= 2.5.1
* CUDA Version >= 12.4

```
conda create -n videomerge python=3.10 -y
conda activate videomerge
pip install -r requirements.txt
```

## Inference
Run the following code to generate a long video. See parse_arg() function in kstep_search.py and config files in ./configs.
```
python long_video_inference.py --num_chunks 7 --resolution 320p --high_freq_merge
```
With default configuration, the model is expected to genereate a 445-frame video of 320p resolution in 22 min.

Put your long prompt in the ./prompt.txt file.

You may change the num_chunks parameters to longer(each chunk is 61 frames).

You may also chanege the resolution to 720p, but it will takes much longer to generate a long video(around 7 hours for num_chunks=5).

