# 读取原始文件所有行
with open('/data1/wqc/project/MinimaxDiffusion-main/MinimaxDiffusion-main/misc/class_indices.txt', 'r') as f:
    lines = f.readlines()

# 检查总行数是否为4的倍数
total = len(lines)
if total % 5 != 0:
    print(f"警告：总行数{total}非4的倍数，实际拆分前{total//4*4}行")
    lines = lines[:total//4*4]

# 按块写入新文件
chunk_size = 200
for i in range(5):
    start = i * chunk_size
    end = start + chunk_size
    chunk = lines[start:end]
    
    filename = f'class_part{i+1}.txt'
    with open(filename, 'w') as f:
        f.writelines(chunk)
    print(f"已生成 {filename}，包含{len(chunk)}行")