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
def extract_and_deduplicate(directory):
    """
    获取指定目录下所有.dot文件名@后的字符串，去重后打印。

    Args:
        directory (str): 目录路径。
    """
    extracted_strings = set()  # 使用集合进行去重
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".dot"):
                match = re.search(r"@(.*)\.dot$", filename)
                if match:
                    extracted_strings.add(match.group(1)) #添加到set中，重复项会自动忽略。
        for s in extracted_strings:
            print(s) #打印set中的所有内容
    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 未找到。请检查路径是否正确。")
    except Exception as e:
        print(f"发生未知错误：{e}")

# 指定目录路径
directory_path = r"C:\Users\86130\Desktop\Code\Pycharm\flaky_test_prediction\data\raw"

# 调用函数
extract_and_deduplicate(directory_path)