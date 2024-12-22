import os
import subprocess
import argparse

def  get_command_line_args():
    parser = argparse.ArgumentParser(description="Process video files with OpenFace landmarks and audio.")
    
    # 添加命令行参数
    parser.add_argument('--process_num', type=int, default=1000, help="num for deciding  to process  videos.")
    parser.add_argument('--model_path', type=str, default='./asserts/training_model_weight/clip_training_256/netG_model_epoch_200.pth', help="num for deciding  to process  videos.")
    parser.add_argument('--audio_path', type=str, default="./asserts/examples/driving_audio_1.wav" , help="path of audio")
    
    # 解析命令行参数
    return parser.parse_args()

if __name__ =="__main__":
    arg= get_command_line_args()
    model_path=arg.model_path
    if not os.path.exists(model_path):
        model_path="./asserts/clip_training_DINet_256mouth.pth"
    
    print(f"model_path: {model_path}")

    audio_path=arg.audio_path
    if not os.path.exists(audio_path):
        audio_path="./asserts/examples/driving_audio_1.wav" 
    print(f"audio_path: {audio_path}")

    # 视频文件夹路径
    video_folder = "./asserts/test_data/split_video_25fps"  # 替换为你的实际视频文件夹路径
    landmark_folder = "./asserts/test_data/split_video_25fps_landmark_openface"  # 替换为你的landmark文件夹路径
    output_folder = "./asserts/inference_result"  # 替换为你希望输出结果的文件夹路径
    #audio_folder = "./asserts/examples/driving_audio_1.wav"  # 替换为你的音频文件夹路径
    audio_folder=audio_path
    pretrained_model_path = model_path # 替换为你的预训练模型路径

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 获取所有的视频文件
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    max=len(video_files)
    num=0

    # 批量处理每个视频文件
    for video_file in video_files:
        # 这里假设输入视频文件与landmark和音频文件具有相同的文件名
        video_name = os.path.splitext(video_file)[0]  # 获取文件名（不带扩展名）

        # 构建输入文件路径
        input_video_path = os.path.join(video_folder, video_file)
        input_landmark_path = os.path.join(landmark_folder, video_name + '.csv')
        input_audio_path = audio_folder

        # 构建命令并调用模型推理的 Python 脚本
        try:
            print(f"Processing video: {video_file}")
            subprocess.run([
                "python", "inference.py",
                "--mouth_region_size", "256",  # 这里的参数根据需求调整
                "--source_video_path", input_video_path,
                "--source_openface_landmark_path", input_landmark_path,
                "--driving_audio_path", input_audio_path,
                "--pretrained_clip_DINet_path", pretrained_model_path,
            ], check=True)
            print(f"Successfully processed: {video_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {video_file}: {e}")
        num+=1
        if num>=arg.process_num or num>max:
            break