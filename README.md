# ML-hw-01
ДЗ1 по курсу "Машинное обучение"
Шильцов Дмитрий Александрович (МОВС ВШЭ 1 год)

Рассматривались варианты реализации линейной регрессии для датасета по б.у автомобилям

Что было сделано

1. Была осуществлены EDA и предобработка данных
   - удаление дублей в обучающей выбоке
   - парсинг столбцов (в том числе разбивка столбца torque на 2 столбца)
   - заполнение пропусков через Imputer
     (важно заполнять пропуски онинаковым образом и на трейне и на тесте)
   - нормализация данных через StandardScaler (помогает лучше учить алгоритм)
   - OHE кодирование категориальных признаков
   - анализ зависимостей признаков посредством графиков

2. Алгоритмы машинного обучения
   - Реализованы алгоримы LenearRegression, Lasso, Ridge, ElasticNet
   - произведен подбор оптимальных значений гиперпараметров
   - наилучший результат дал Ridge (но результат не очень впечатляющий)

3. Реализация сервиса через FastApi
   - сервис реализован локально, протестирован на одиночных объектах и на массиве объектов (скриншоты прилагаю)
   - веса модели а также параметры энкодера, импьютера и скалера  лежат в папке models
    
4. Дополнительные задачи
   - нарисован график зависимости цены от числа сидений (в EDA) - для подтверждения нелинейной зависимости
   - графики зависимостей метрики R2 от значений гиперпараметров для Lasso и Ridge
   - график совместного распределения пробега/года выпуска в срезе типа топлива и сделаны выводы
   - построена линейная регрессия с L1 регуляризацией на полиномиальных признаках степени 2,
     по результатам видно что все категориальные признаки оказалить  не такими и важными,
     при этом модель выдала существенно лучшее качество на кросс-валидации чем все предыдущие
     (R2 = 0.758)
   
      


Картинки из постмана для локально развернутого FastAPI:

![изображение](https://github.com/shiltsov/ML-hw-01/assets/54742337/774a75b3-47bb-42bf-8636-783ab5aa6a30)
![изображение](https://github.com/shiltsov/ML-hw-01/assets/54742337/6bb952ad-b68e-4624-b61c-1f3d8861b765)
