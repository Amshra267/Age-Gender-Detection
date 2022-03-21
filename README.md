# BOSCH_A-G_INTERIIT

Repository containing **TEAM-IITBHU** Solution for BOSCH Age and Gender Detection Problem Statement in InterIIT tech meet 10.0

## INSTALLATION

**Step1** - Cloning this Repository

 - In terminal/git bash run 

 ```bash
 git clone https://github.com/Amshra267/BOSCH_A-G_INTERIIT.git
 ## move to that directory
 mkdir BOSCH_A-G_INTERIIT
 ```  

**Step 2** - We had provided two configurations for installation

 - **For GPU**(Recommended) - (Requirements -  **```CUDA version = 11.1```**)
    
     * In the above opened terminal run

     ```bash
     chmod +x install_gpu.bash
     ./install_gpu.bash
     ```

 - **For CPU**
    
     * In the above opened terminal run
     
     ```bash
     chmod +x install_cpu.bash
     ./install_cpu.bash
     ```

Now you are ready to run our solution

## TESTING

In terminal run 

```python main.py --video <path_to_your_video_file>```

Above file have some arguments:-

    --video = <path of video file> 
                 or <camera id_no in case of real time image feed
                example = 0 (for webcam ), 1 (for external camera) >
    
    --show  = True (for showing our output), default = false  

## References
 - Yolov4 (For Object Detection) - https://arxiv.org/abs/2004.10934 
 - Deep-Sort (For Object Tracking) - https://arxiv.org/abs/1703.07402
 - ESRGAN (For super resolution) - https://arxiv.org/abs/1809.00219
 - UNet (For Segmentation Mask) - https://arxiv.org/abs/1505.04597
 - Gait Recognition (For Age and Gender analysis) - https://link.springer.com/10.1007/978-1-4419-5906-5_741