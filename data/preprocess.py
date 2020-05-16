
path_from = "origin_data\\mscoco\\"
path_to = "mscoco_data\\"

with open(path_from + 'train_target.txt', 'r') as fileread:
    lines = fileread.readlines()

total = len(lines)
print(total)
print(type(lines[10]))

train_lines = lines[:int(total*0.8)]
val_lines = lines[int(total*0.8):]

print(len(train_lines), len(val_lines))

with open(path_to + 'train_target.txt', 'w') as filewrite:
    filewrite.writelines(train_lines)

with open(path_to + 'val_target.txt', 'w') as filewrite:
    filewrite.writelines(val_lines)



