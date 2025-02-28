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

# Столбцы которые являются строки 
string_cols = data.select_dtypes(include='object').columns
print("Столбцы со строковым типом данных:", list(string_cols))


# Уникальных значений в каждом строковом столбце
for col in string_cols:
    print(f"\nСтолбец: {col}")
    print("Уникальные значения:", data[col].unique()[:5]) 