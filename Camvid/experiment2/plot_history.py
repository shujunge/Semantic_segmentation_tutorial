import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
base_paths ="./training_history"
imagescales=["128_128","256_256"]
loss_name = "dice_coef_loss"
model_names=["unet","segnet"]

name_lists=[os.path.join(base_paths,index) for index in os.listdir(base_paths)]



plot_data_valloss=[]
plot_data_valIOU=[]
plot_data_loss=[]
plot_data_IOU=[]

index =[]

for model_name in model_names:
  for imagescale in imagescales:
    name = imagescale+model_name + "_" + loss_name
    temp = pd.read_csv("./training_history/%s_history.csv" % (name))
    plot_data_valloss.append(temp.iloc[:, 0].tolist())
    plot_data_valIOU.append(temp.iloc[:, 1].tolist())
    plot_data_loss.append(temp.iloc[:, 2].tolist())
    plot_data_IOU.append(temp.iloc[:, 3].tolist())
    index.append(name)

total_data=[]
total_data.append(plot_data_valloss)
total_data.append(plot_data_valIOU)
total_data.append(plot_data_loss)
total_data.append(plot_data_IOU)



## visualize the results
label_list = index.copy()
del index

color_list = ['r', 'b', 'g', 'y', 'k', 'm', 'c', '345']
title_names =['val_loss','val_IOU','loss','IOU']


for my_index in range(4):
  showing_data=total_data[my_index]
  plt.figure(figsize=(8,8))


  for data_index in range(len(showing_data)):
    plt.plot(["%d"%i for i in range(len(showing_data[data_index]))],showing_data[data_index],
    color=color_list[data_index],label=label_list[data_index])
    
  ticks =["%d"%i for i in range(0,120,15)]
  
  plt.xticks(ticks, ticks) #rename the x axis
  if my_index%2==1:
    plt.legend(loc='lower right') #lower right,upper right
  else:
    plt.legend(loc='upper right')
  plt.title("compare different parameters of UNet and SegNet")
  plt.xlabel("epochs")
  plt.ylabel(title_names[my_index])
  plt.savefig("../summary/model_%s.png"%title_names[my_index],dpi=560)
  plt.show()








