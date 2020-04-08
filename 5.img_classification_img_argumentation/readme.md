Image classifier using tensorflow Image ImageDataGenerator class +  Image Argumentation Parameters
---------------------------------------------------------------------------------------------------
1.img_argu.py :- Model Creation, Model Compilation, Model test

2.directory('dataset' ) :-  Training dataset for model

---------------------------------------------------------------------------------------------------

Code sample :-

train_data_gen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)


--------------------------------------------------------------------------------------------------

Refrence :-

https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb#scrollTo=BZSlp3DAjdYf

