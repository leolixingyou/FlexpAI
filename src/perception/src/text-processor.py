def process_file(input_filename, output_filename, n):
    """
    从文件中读取数据，横向每两个值为一组，取前n组的第一个值
    
    参数:
    input_filename: 输入文件名
    output_filename: 输出文件名
    n: 要处理的组数
    """
    results = []
    
    try:
        # 读取输入文件
        with open(input_filename, 'r') as file:
            data = file.readlines()
        data_temp =  [x for i,x in enumerate(data) if i % n ==0 ]
        # 将结果写入输出文件
        with open(output_filename, 'w') as outfile:
            for value in data_temp:
                outfile.write(f"{value}\n")
                
        print(f"处理完成！结果已写入到 {output_filename}")
                    
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_filename}")
    except Exception as e:
        print(f"处理文件时出错：{e}")
        
    return results

# 使用示例
if __name__ == "__main__":
    filename = "/workspace/src/control/src/example/odm_x_y_yaw_abs_log_8.txt"
    output_filename = "/workspace/src/control/src/example/temp.txt"
    n = int(input("请输入缩放倍数(n): "))
    
    results = process_file(filename, output_filename, n)
    
    print("\n处理结果预览:")
    for i, value in enumerate(results, 1):
        print(f"第{i}行的结果: {value}")