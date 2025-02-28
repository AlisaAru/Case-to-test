import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Первое задание анализ данных
# Тут можно вривести базовые манипуляции с данными: сбор данных
# привести данные в читабельный вид, удалить дубликаты, привести в один регистр
# Можно сделать визуализацию и может примерную статистику сделать

# У меня данные через каггл и в типе данных csv
# исползую пандас чтобы его прочесть 

data = pd.read_csv("labelled_newscatcher_dataset.csv", sep=";")
# У меня данные в коллонках сохранены через ;, поэтому чтобы прочитать надо юзать sep=";"

# Отображение данных 
print("Показать первые строки сверху: ")
print(data.head())
print()

# Информация по базе
print("Информация о базе:")
data.info()
print()

# Найти все пустые значения
print("Отображение суммы пустых значений: ")
print(data.isnull().sum())
print()

# Увидеть статистику
print("Увидеть статистику:")
print(data.describe())
print()


# Если хочешь узнать колличество отпрделенного элементов через название колонки 
if 'lang' in data.columns:
    print("\nLabel Distribution:")
    print(data['lang'].value_counts())

# Чтобы мы могли выделить данные по их типу и их отобразить, можно использовать сортировку 
num_cols = data.select_dtypes(include='object').columns

if len(num_cols) == 0:
    print("No numeric columns to visualize.")
elif len(num_cols) == 1:
    # If only one numeric column, plot a histogram
    col = num_cols[0]
    plt.hist(data[col].dropna(), bins='auto', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.show()
else:
    # If multiple numeric columns, show a correlation matrix
    corr = data[num_cols].corr()
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation Matrix")
    plt.show()
