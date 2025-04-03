import os

def count_subdirectories(path):
    try:
        # 列出路径下的所有文件和文件夹，并筛选出子文件夹
        subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return len(subdirectories)
    except FileNotFoundError:
        return "路径不存在，请检查路径是否正确。"
    except Exception as e:
        return f"发生错误: {e}"

# 替换为实际路径
path = "node/Transformers-main/models"
print(f"子文件夹数量: {count_subdirectories(path)}")
