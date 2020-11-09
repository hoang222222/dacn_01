import numpy as np
import csv
import matplotlib.pyplot as plt


#doc du lieu
with open('hand_written.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []

    # đọc từng dòng của file và thêm vào list rows, mỗi phần tử của list là một dòng
    for row in result:
        rows.append(row)

letter = rows[30000]
x = np.array([int(j) for j in letter[1:]])
x = x.reshape(28, 28)

print(letter)
plt.imshow(x, cmap="Greys", interpolation= None)
plt.show()