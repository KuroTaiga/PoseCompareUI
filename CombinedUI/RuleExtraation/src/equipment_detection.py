import sys
import os
import logging
import subprocess

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_equipment_via_script(detect_script_path, weights_path, video_path, conf_threshold=0.4, img_size=640):
    """通过外部脚本检测视频中的器材"""
    try:
        # 验证文件路径
        for path, name in [
            (detect_script_path, "Detection script"),
            (weights_path, "Weights file"),
            (video_path, "Video file")
        ]:
            if not os.path.exists(path):
                logging.error(f"{name} not found at: {path}")
                return "none"
            logging.info(f"{name} found at: {path}")

        # 构建命令
        cmd = [
            sys.executable,  # 使用当前Python解释器
            detect_script_path,
            '--weights', weights_path,
            '--source', video_path,
            '--conf-thres', str(conf_threshold),
            '--img-size', str(img_size),
            '--nosave',
            '--device', '0'  # 指定GPU设备
        ]
        
        logging.info(f"Executing command: {' '.join(cmd)}")

        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
        
        # 运行命令并捕获输出
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=os.path.dirname(detect_script_path),  # 设置工作目录
                check=True  # 这将导致非零返回码抛出异常
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with return code {e.returncode}")
            logging.error(f"Standard output:\n{e.stdout}")
            logging.error(f"Standard error:\n{e.stderr}")
            return "none"

        # 如果有警告信息，记录下来
        if result.stderr:
            logging.warning(f"Detection warnings:\n{result.stderr}")

        # 解析输出
        equipment_str = "none"
        lines = result.stdout.split('\n')
        logging.debug(f"Full output:\n{result.stdout}")
        
        for i, line in enumerate(lines):
            if "=== Final Detected Equipment ===" in line:
                if i + 1 < len(lines):
                    equipment_dict_str = lines[i + 1].strip()
                    try:
                        equipment_dict = eval(equipment_dict_str)
                        if equipment_dict:
                            # 添加置信度阈值判断
                            max_equipment = max(equipment_dict, key=equipment_dict.get)
                            max_confidence = equipment_dict[max_equipment]
                            
                            # 只有当置信度超过阈值时才接受检测结果
                            if max_confidence > 0.5:  # 设置更高的置信度阈值
                                equipment_str = max_equipment
                            else:
                                equipment_str = "none"
                                logging.info(f"Equipment {max_equipment} detected but confidence {max_confidence} too low")
                        else:
                            equipment_str = "none"
                    except Exception as e:
                        logging.error(f"Error parsing equipment dict: {e}")
                        logging.error(f"Raw equipment string: {equipment_dict_str}")
                        equipment_str = "none"
                break

        logging.info(f"Detected equipment: {equipment_str}")
        return equipment_str

    except Exception as e:
        logging.error(f"Error in equipment detection: {str(e)}", exc_info=True)
        return "none"

# 测试函数
def test_detection():
    """测试器材检测功能"""
    # 设置测试参数
    BASE_PATH = '/content/drive/MyDrive/rez'
    YOLO_SCRIPT = os.path.join(BASE_PATH, 'detect_revise.py')
    YOLO_WEIGHTS = os.path.join(BASE_PATH, 'best.pt')
    VIDEO_PATH = os.path.join(BASE_PATH, 'rule_test_video')

    # 获取第一个测试视频
    test_videos = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    if not test_videos:
        logging.error(f"No test videos found in {VIDEO_PATH}")
        return
    
    test_video = os.path.join(VIDEO_PATH, test_videos[0])
    logging.info(f"Testing with video: {test_video}")

    # 执行检测
    result = detect_equipment_via_script(YOLO_SCRIPT, YOLO_WEIGHTS, test_video)
    logging.info(f"Detection result: {result}")

if __name__ == "__main__":
    test_detection()