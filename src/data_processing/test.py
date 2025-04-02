import os
import re
import os

# def rename_async_wait_files(directory):
#     """
#     重命名指定目录下所有以 "async wait .dot" 结尾的文件，将其修改为 "async wait.dot"。
#
#     参数:
#     directory (str): 目录路径。
#     """
#     try:
#         for filename in os.listdir(directory):
#             if filename.endswith("test case timeout .dot"):
#                 new_filename = filename.replace("test case timeout .dot", "test case timeout.dot")
#                 old_filepath = os.path.join(directory, filename)
#                 new_filepath = os.path.join(directory, new_filename)
#                 os.rename(old_filepath, new_filepath)
#                 print(f"将 {filename} 重命名为 {new_filename}")
#     except FileNotFoundError:
#         print(f"错误：目录 {directory} 未找到。")
#     except Exception as e:
#         print(f"发生错误：{e}")
#
# # 指定目录路径
# directory_path = r"C:\Users\86130\Desktop\Code\Pycharm\flaky_test_prediction\data\raw"
#
# # 执行重命名
# rename_async_wait_files(directory_path)
#
# print("操作完成。")
import os
import re
import shutil


def extract_and_deduplicate_and_copy(directory, destination):
    """
    获取指定目录下所有.dot文件名@后的字符串，去重后打印，并统计每个类别下的文件个数，
    将统计到超过30个文件的类别下所有文件复制到目标文件夹。

    Args:
        directory (str): 源目录路径。
        destination (str): 目标目录路径。
    """
    category_files = {}  # 用于记录每个类别对应的文件列表
    try:
        # 遍历目录下所有文件
        for filename in os.listdir(directory):
            if filename.endswith(".dot"):
                match = re.search(r"@(.*)\.dot$", filename)
                if match:
                    category = match.group(1)
                    category_files.setdefault(category, []).append(filename)

        # 打印每个类别及对应的文件个数
        for category, files in category_files.items():
            count = len(files)
            print(f"{category}: {count}个文件")

        # 创建目标目录（如果不存在）
        os.makedirs(destination, exist_ok=True)

        # 对于每个类别，如果文件数超过30，将这些文件复制到目标目录
        for category, files in category_files.items():
            if len(files) > 35:
                for filename in files:
                    src_path = os.path.join(directory, filename)
                    dest_path = os.path.join(destination, filename)
                    shutil.copy(src_path, dest_path)
                print(f"类别 '{category}' 的 {len(files)} 个文件已复制到 '{destination}'")

    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 未找到。请检查路径是否正确。")
    except Exception as e:
        print(f"发生未知错误：{e}")


# 指定源目录和目标目录路径
source_directory = r"C:\Users\86130\Desktop\Code\Pycharm\flaky_test_prediction\data\raw"
destination_directory = r"C:\Users\86130\Desktop\Code\Pycharm\flaky_test_prediction\data\raw_2"

# 调用函数
extract_and_deduplicate_and_copy(source_directory, destination_directory)

