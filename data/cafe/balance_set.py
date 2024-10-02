import os
import shutil

tappy_set = [7, 8, 24, 25, 41, 42, 60, 61, 77, 78, 94, 95, 112, 113, 128, 129, 147, 148,
             164, 165, 181, 182, 187, 198, 199, 217, 218, 236, 237, 255, 256, 274, 275, 293,
             294, 312, 313, 331, 332, 350, 351, 369, 370, 388, 389, 407, 408, 426, 427,
             446, 447, 465, 466, 484, 485, 503, 504, 522, 523, 541, 542, 560, 561, 579,
             580, 598, 599, 617, 618, 636, 637, 655, 656, 674, 675, 693, 694, 712, 713,
             731, 732, 750, 751, 769, 770, 788, 789, 807, 808, 826, 827, 845, 846, 864, 865,
             883, 884, 902, 903, 921, 922, 940, 941, 959, 960, 978, 979, 997, 998, 1016, 1017,
             1035, 1036, 1054, 1055, 1071, 1072, 1090, 1091, 1109, 1110, 1129, 1147, 1148, 1166, 1167,
             1185, 1186, 1204, 1205, 1223, 1224, 1242, 1243, 1261, 1262]  # extract the NO. of all teeth showing face
happy_set = [i-2 for i in tappy_set]
happy_set.append(1120)
happy_set.append(1128)
tappy_balance = [i for i in tappy_set if i & 1 == 0]
happy_balance = [i for i in happy_set if i & 1 != 0]  # extract the NO. of half happy face

happy_combination = happy_balance + tappy_balance

sad_set = [9, 10, 26, 27, 43, 44, 62, 63, 130, 131, 149, 150, 166, 167, 183, 184, 200, 201, 202,
           219, 220, 238, 239, 257, 258, 276, 277, 295, 296, 314, 315, 333, 334, 352, 353,
           371, 372, 390, 391, 409, 410, 428, 429, 448, 449, 467, 468, 486, 487, 505, 506,
           524, 525, 543, 544, 562, 563, 581, 582, 600, 601, 619, 620, 638, 639, 657, 658,
           676, 677, 695, 696, 714, 715, 733, 734, 752, 753, 771, 772, 790, 791, 809, 810,
           866, 867, 885, 886, 904, 905, 923, 924, 942, 943, 961, 962, 980, 981, 999, 1000, 1018, 1019,
           1037, 1038, 1056, 1057, 1073, 1074, 1092, 1093, 1111, 1112, 1130, 1131, 1149, 1150, 1168,
           1169, 1187, 1188, 1206, 1207, 1225, 1226, 1244, 1245, 1263, 1264]  # extract the NO. of all sad face
anger_set = [1, 2, 18, 19, 35, 36, 52, 53, 71, 72, 88, 89, 106, 107, 122, 123, 139, 140,
             158, 159, 175, 176, 192, 193, 209, 210, 228, 229, 247, 248, 266, 267, 285, 286,
             304, 305, 323, 324, 342, 343, 361, 362, 380, 381, 399, 400, 418, 419, 437, 438,
             457, 458, 476, 477, 495, 496, 514, 515, 533, 534, 552, 553, 571, 572, 590, 591,
             609, 610, 628, 629, 647, 648, 666, 667, 685, 686, 704, 705, 723, 724, 742, 743,
             761, 762, 780, 781, 799, 800, 818, 819, 837, 838, 856, 857, 875, 876, 894, 895,
             913, 914, 932, 933, 951, 952, 970, 971, 989, 990, 1008, 1009, 1027, 1028, 1046,
             1047, 1065, 1066, 1082, 1083, 1101, 1102, 1121, 1122, 1139, 1140, 1158, 1159, 1177,
             1178, 1196, 1197, 1215, 1216, 1234, 1235, 1253, 1254]  # extract the NO. of all angry face
disgusted_set = [3, 4, 20, 21, 37, 38, 54, 55, 73, 74, 90, 91, 108, 109, 124, 125, 141, 142,
                 160, 161, 177, 178, 194, 195, 211, 212, 230, 231, 249, 250, 268, 269, 287, 288,
                 306, 307, 325, 326, 344, 345, 363, 364, 382, 383, 401, 402, 420, 421, 439, 440,
                 441, 459, 460, 478, 479, 497, 498, 516, 517, 535, 536, 554, 555, 573, 574, 592,
                 593, 611, 612, 630, 631, 649, 650, 668, 669, 687, 688, 706, 707, 725, 726, 744,
                 745, 763, 764, 782, 783, 801, 802, 820, 821, 839, 840, 858, 859, 877, 878, 896,
                 897, 915, 916, 934, 935, 953, 954, 972, 973, 991, 992, 1010, 1011, 1029, 1030,
                 1048, 1049, 1067, 1068, 1084, 1085, 1103, 1104, 1123, 1124, 1141, 1142, 1160, 1161,
                 1179, 1180, 1198, 1199, 1217, 1218, 1236, 1237, 1255, 1256]  # extract the NO. of all disgusted face
fear_set = [16, 34, 50, 56, 57, 143, 144, 213, 214, 232, 233, 251, 252, 270, 271, 289, 290, 308, 309, 327, 328, 346, 347, 365, 366,
            384, 385, 403, 404, 422, 423, 442, 443, 461, 462, 480, 481, 499, 500, 518, 519,
            537, 538, 556, 557, 575, 576, 594, 595, 613, 614, 632, 633, 651, 652, 670, 671,
            689, 690, 708, 709, 727, 728, 746, 747, 765, 766, 784, 785, 803, 804, 822, 823,
            841, 842, 860, 861, 879, 880, 898, 899, 917, 918, 936, 937, 955, 956, 974, 975,
            993, 994, 1012, 1013, 1031, 1032, 1050, 1051, 1086, 1087, 1105, 1106, 1125, 1126,
            1143, 1144, 1162, 1163, 1181, 1182, 1200, 1201, 1219, 1220, 1238, 1239, 1257,
            1258]  # extract the NO. of all feared face
sup_set = [17, 33, 51, 69, 70, 86, 87, 103, 104, 137, 138, 156, 157, 173, 174,
           190, 191, 207, 208, 226, 227, 245, 246, 264, 265, 283, 284, 302, 303, 321, 322,
           340, 341, 359, 360, 378, 379, 397, 398, 416, 417, 435, 436, 455, 456, 474, 475,
           493, 494, 512, 513, 531, 532, 550, 551, 569, 570, 588, 589, 607, 608, 626, 627,
           645, 664, 665, 683, 684, 702, 703, 721, 722, 740, 741, 759, 760, 778, 779, 797,
           798, 816, 817, 835, 836, 854, 855, 873, 874, 892, 893, 911, 912, 930, 931, 949,
           950, 968, 969, 987, 988, 1006, 1007, 1025, 1026, 1044, 1045, 1063, 1064, 1080, 1081,
           1099, 1100, 1118, 1119, 1137, 1138, 1156, 1157, 1175, 1176, 1194, 1195, 1213, 1214,
           1232, 1233, 1251, 1252, 1270, 1271, 1272]  # extract the NO. of all surprised face, 1272 is a duplication

neutral_set = [i for i in range(1, 1273) if i not in sup_set + fear_set + disgusted_set + tappy_set + happy_set + anger_set + sad_set]
neutral_balance = [i for i in neutral_set if i & 1 == 0]

unselected_set = [i for i in range(1, 1273) if i not in sup_set + fear_set + disgusted_set + happy_combination + anger_set + sad_set + neutral_balance]

# os.mkdir('balance_all')
# for i in os.listdir('full_data'):
# 	if os.path.splitext(i)[1] == '.jpg':
# 		if int(i.split('.')[0]) in sup_set + fear_set + disgusted_set + anger_set + sad_set + happy_combination + neutral_balance:
# 			old = 'full_data/' + i
# 			new = 'balance_all/' + i
# 			shutil.copyfile(old, new)

# os.mkdir('/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/unselect_set')
# for i in os.listdir('/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/full_data'):
#   if os.path.splitext(i)[1] == '.jpg':
#       if int(i.split('.')[0]) in unselected_set:
#           old = '/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/full_data/' + i
#           new = '/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/unselect_set/' + i
#           shutil.copyfile(old, new)

# os.mkdir('/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/happy')
# for i in os.listdir('/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/full_data'):
#   if os.path.splitext(i)[1] == '.jpg':
#       if int(i.split('.')[0]) in happy_combination:
#           old = '/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/full_data/' + i
#           new = '/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/happy/' + i
#           shutil.copyfile(old, new)

# os.mkdir('/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/sad')
# for i in os.listdir('/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/full_data'):
#   if os.path.splitext(i)[1] == '.jpg':
#       if int(i.split('.')[0]) in sad_set:
#           old = '/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/full_data/' + i
#           new = '/Users/Yan/Projects/experiments/study1_VAEguidedMCMCP/VAE/dataset/FaceImages/sad/' + i
#           shutil.copyfile(old, new)

def read_label(filepath):
    """read label.csv and return a list"""

    file = open(filepath, 'r', encoding="utf-8")
    context = file.read()  # as str
    list_result = context.split("\n")[1:-1]
    length = len(list_result)

    for i in range(length):
        list_result[i] = int(list_result[i].split(",")[-1])

    file.close()  # file must be closed after manipulating
    return list_result

# labels = read_label('balance_all/label.csv')
labels_identity = read_label('balance_all/label_identity.csv')

os.mkdir('neutral_set')
filenames = [file for file in os.listdir('balance_all') if os.path.splitext(file)[-1] == '.jpg']
filenames.sort(key=lambda x: int(x.split('.')[0])) 

for n, i in enumerate(filenames):
  if int(i.split('.')[0]) in neutral_set:
    old = 'balance_all/' + i
  else:
    for j in range(len(labels_identity)):
      if labels_identity[j] == labels_identity[n] and int(filenames[j].split('.')[0]) in neutral_set:
        old = 'balance_all/' + filenames[j]
        break

  new = 'neutral_set/' + i
  shutil.copyfile(old, new)
