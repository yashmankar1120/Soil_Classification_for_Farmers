import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models,layers
w = tk.Tk()
w.geometry("1200x700")
w.title("Main Window")
w.configure(bg='#FFC872')
sign_image = Label(w,bg='#FFC872')
file_path=""
EPOCHS=1
history=''

def upload_image():
    global resize_image, file_path

    try:

        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        resize_image = uploaded.resize((300, 205))

        im = ImageTk.PhotoImage(resize_image)
        sign_image.configure(image=im)
        sign_image.image = im
    except:
        pass

def detect_soil():
    global EPOCHS, history, output


    BATCH_SIZE = 30
    IMAGE_SIZE = 256
    EPOCHS = 1
    CHANNELS = 3
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "Soil-Dataset", seed=123, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(class_names)
    print(len(dataset))

    for image_batch, label_batch in dataset.take(1):
        print(image_batch.shape)
        print(image_batch[1])
        print(label_batch.numpy())

    plt.figure(figsize=(15, 15))
    for image_batch, labels_batch in dataset.take(1):
        for i in range(BATCH_SIZE):
            ax = plt.subplot(8, 8, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels_batch[i]])
            plt.axis("off")

    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        # Autotune all the 3 datasets
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1. / 255),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = 9

    model = models.Sequential([
        resize_and_rescale,
        # data_augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.build(input_shape=input_shape)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        verbose=1,
        epochs=EPOCHS,
    )


    model.evaluate(test_ds)

    acc = history.history['accuracy']
    loss = history.history['loss']
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(EPOCHS), acc, label=' Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(EPOCHS), loss, label=' Loss')
    # plt.legend(loc='upper right')
    # plt.title('Loss')
    # plt.show()

    # image_path = "Soil-Dataset/Black Soil/6.jpg"

    image = preprocessing.image.load_img(file_path)
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    print(resize_image)

    pred = model.predict(scaled_img)
    output = class_names[np.argmax(pred)]
    print(output)
    Label(w, text=output, width=12, height=2, font=('Arial',12,'bold'),bg='#FFE3B3').place(x=275, y=378)

def summary():
    if output == 'Alluvial Soil':
        Label(w, text="1.REGION: Nothern Plains, Assam, Bihar and West Bengal",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=460)
        Label(w, text="2.CHARACTERISTICS: Rich in Humus and organic matter and Phosphoric Acid.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS: Cotton, Wheat, Sorghum, Bajra, Maize",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50,y=540)
        Label(w, text="4.FERTILIZER: 1.Manure  2.Compost  3.Fish Extract",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50,y=580)
        Label(w, text="5.WATER SUPPLY: 75cm to 100cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE: 21`C to 28`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)

    elif output == "Black Soil":
        Label(w, text="1.REGION: Gujarat, Madhya Pradesh, Maharashtra, Andhra Pradesh,Tamil Nadu, and Telangana",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50,y=460)
        Label(w, text="2.CHARACTERISTICS: Rich in magnesium, iron, aluminum, and lime.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS: Cotton,Wheat,Linseed,Oilseeds",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=540)
        Label(w, text="4.FERTILIZER: 1.Cocpeat  2.Vermicompost",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=580)
        Label(w, text="5.WATER SUPPLY: 60cm to 80cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE: 27`C to 32`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)

    elif output == 'Red Soil':
        Label(w, text="1.REGION: Deccan Plateau",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=460)
        Label(w, text="2.CHARACTERISTICS: Rich in Potash and is somewhat Acidic in nature.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS: Groundnut, Potato, Maize(Corn), Rice, Ragi, Wheat, Millets, Pulses",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=540)
        Label(w, text="4.FERTILIZER: 1.Ammonium Sulphate",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=580)
        Label(w, text="5.WATER SUPPLY: 140cm to 200cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE: 18`C to 28`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)


    elif output == 'Yellow Soil':
        Label(w, text="1.REGION:Middle Ganga plain and Piedmont zone of Western Ghats",bg='#FFE3B3',font=('Arial',8,'bold'),).place(x=50, y=460)
        Label(w, text="2.CHARACTERISTICS:Rich in Iron Oxides.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS:Groundnut, Potato, Cofee, Coconut,Rice etc.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=540)
        Label(w, text="4.FERTILIZER: 1.Triple Super Phosphate",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=580)
        Label(w, text="5.WATER SUPPLY: 25cm to 60cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE: 20`C to 25`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)


    elif output == 'Laterite Soil':
        Label(w, text="1.REGION: Central India and Western Peninsula.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=460)
        Label(w, text="2.CHARACTERISTICS: It is Acidic in nature and is not very fertile.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS: Cotton, Wheat, Rice, Pulses, Rubber, Tea, Coffee, Coconut, and Cashews.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=540)
        Label(w, text="4.FERTILIZER: 1.Sodium Silicate",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=580)
        Label(w, text="5.WATER SUPPLY: 125cm to 200cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE:21`C to 30`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)

    elif output == 'Arid Soil':
        Label(w, text="1.REGION: Haryana, Western Rajasthan, Punjab and the Rann of Kutch",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=460)
        Label(w, text="2.CHARACTERISTICS: Sandy texture and quick draining in nature.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS: Corn, Sorghum, Pearl Millets, Seasame.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50,y=540)
        Label(w, text="4.FERTILIZER: 1.Ammonium Nitrate  2.Ammonium Phosphate",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50,y=580)
        Label(w, text="5.WATER SUPPLY: 50cm to 75cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE: 20`C to 30`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)

    elif output == 'Mountain_Forest Soil':
        Label(w, text="1.REGION: Western/Eastern Ghats and a few regions of the Peninsular Plateau.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=460)
        Label(w, text="2.CHARACTERISTICS: Rich in Humus and organic Matter.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=500)
        Label(w, text="3.SUITABLE CROPS: Maize, Tea, Coffee, Spices, Tropical and Temperate fruits.",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=540)
        Label(w, text="4.FERTILIZER: 1.Ammonium Nitrate",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=580)
        Label(w, text="5.WATER SUPPLY: 50cm to 75cm",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=620)
        Label(w, text="6.TEMPERATURE: 20`C to 30`C",font=('Arial',8,'bold'),bg='#FFE3B3').place(x=50, y=660)

def accuracy():
    Label(w,text='95%',font=('Arial',10,'bold'),width=10,height=2).place(x=490,y=390)

def accuracy_graph():
    acc = history.history['accuracy']
    loss = history.history['loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label=' Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label=' Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()




Label(w,text='Soil Classification Using Deep Learning',font=('Arial',18,'bold','underline'),bg='#FFC872',fg='#752092').pack()
Label(w,text='',width=30,height=25,bg='#FFE3B3').place(x=20,y=50)
Label(w,text='Menu',font=('Arial',8,'bold'),bg='#752092',fg='white').place(x=20,y=30)
Button(w,text='Load Soil Image',width=15,font=('Arial',14),fg='white',bg='#752092',command=upload_image).place(x=40,y=65)
Button(w,text='Detect Soil',width=15,font=('Arial',14),fg='white',bg='#752092',command=detect_soil).place(x=40,y=115)
Button(w,text='CNN Accuracy',width=15,font=('Arial',14),fg='white',bg='#752092',command=accuracy).place(x=40,y=165)
Button(w,text='SVM Accuracy',width=15,font=('Arial',14),fg='white',bg='#752092').place(x=40,y=215)
Button(w,text='Best Precision',width=15,font=('Arial',14),fg='white',bg='#752092').place(x=40,y=265)
Button(w,text='Best Sensitivity',width=15,font=('Arial',14),fg='white',bg='#752092').place(x=40,y=315)
Button(w,text='Graph',width=15,font=('Arial',14),bg='#752092',fg='white',command=accuracy_graph).place(x=40,y=365)

Label(w,text='',width=25,height=4,bg='#FFE3B3').place(x=250,y=365)
Label(w,text='Type of Soil',font=('Arial',8,'bold'),bg='#752092',fg='white').place(x=250,y=350)
Label(w,text='Soil Summary',font=('Arial',10,'bold'),bg='#752092',fg='white').place(x=43,y=440)
Label(w,text='',width=160,height=15,bg='#FFE3B3').place(x=43,y=460)
Label(w,text='',width=80,height=4,bg='#FFE3B3').place(x=450,y=365)
Label(w,text='Result',font=('Arial',8,'bold'),bg='#752092',fg='white').place(x=450,y=345)
Label(w,text='CNN Accuracy',font=('Arial',8,'bold','underline'),fg='#752092',width=18,height=2,bg='#FFE3B3').place(x=470,y=365)
Label(w,text='SVM Accuracy',font=('Arial',8,'bold','underline'),fg='#752092',width=18,height=2,bg='#FFE3B3').place(x=600,y=365)
Label(w,text='Best Precision',font=('Arial',8,'bold','underline'),fg='#752092',width=18,height=2,bg='#FFE3B3').place(x=730,y=365)
Label(w,text='Best Sensitivity',font=('Arial',8,'bold','underline'),fg='#752092',width=18,height=2,bg='#FFE3B3').place(x=860,y=365)
Button(w,text='View Soil Summary',font=('Arial',12,'bold'),bg='#752092',fg='white',height=2,command=summary).place(x=1030,y=365)

# upload = Button(w, text="Upload an image", command=upload_image, padx=10, pady=5)
# upload.place(x=20,y=20)
sign_image.place(x=350,y=100)

w.mainloop()