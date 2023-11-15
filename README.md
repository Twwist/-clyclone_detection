<div align="center">
  <h1>Cyclone_detection</h1>
  <p>Проект для Сириус ИИ 2023, задача заключалась в детекции тропических циклонов на спутниковых снимках</p>
</div>
<br>

### Содержание:
- [🚀Введение](#введение)
  
- [⏳Актуальность решения задачи](#введение)
  
- [🎯 Цель и задачи работы](#цели)
  
- [✏️Анализ области детекции циклонов](#анализ)
  
- [📝Формулировка критериев для определения лучшего алгоритма](#формулировка)
  
- [📊Сравнительный анализ алгоритмов согласно критериям](#сравнение)
	-	 [Сверточные нейронные сети (CNN)](#cnn)
	-	[R-CNN,  Fast R-CNN и Faster R-CNN](#rcnn)
	-	[Модели YOLO (You Only Look One Look)](#yolo):
          
- [✏️Описание алгоритма для решения задачи детекции циклонов](#описание)

- [📑Документация по файлам](#документация)

<br>
<br>

<div align="center">
  <h1>Введение</h1>
</div>

Циклоны — это крупномасштабные атмосферные явления, характеризующиеся вращением воздушных масс вокруг области низкого атмосферного давления. Они возникают вследствие неравномерного нагревания Земли и движения воздушных масс по принципу переноса избытка тепла из тропиков в полюса. 

Циклоны формируются вблизи границ атмосферных фронтов и могут быть различной интенсивности — от слабых до разрушительных. Они обладают спиральным вращением, при котором воздух поднимается в центре и снижается по периферии циклона. 

Циклоны могут приносить с собой сильные ветра, выпадение осадков и другие погодные аномалии.

<br>
<br>

<div align="center">
  <h1>Актуальность решения задачи</h1>
</div>

- На сколько решение задачи детекции циклонов полезна и актуальна для нашего времени
  
- Какие проблемы можно решить с её помощью

1. Безопасность населения (Циклоны могут вызвать сильные ветры, обильные дожди и наводнения, что ведет к эвакуации и потерям человеческих жизней)
   
2. Экономические потери (Циклоны могут нанести значительный ущерб экономике, разрушая инфраструктуру, сельское хозяйство и транспорт)
   
3. Исследования изменений климата (Повышение активности тропических циклонов связывается с изменениями климата)

<br>
<br>

<div align="center">
  <h1>🎯 Цель и задачи работы</h1>
</div>

Цели:

1) Разработка и реализация системы детекции тропических циклонов с последующим отслеживанием их движения и классификацией на различные категории.
  
2) Улучшение способности предсказания и мониторинга циклонов с высокой точностью и в реальном времени.
 

Задачи исследования: 
1) `Разработка модели детекции` (Разработать алгоритмы машинного обучения, способные определять наличие тропических циклонов на изображениях; Исследовать и выбрать подходящую архитектуру нейронной сети)
   
2) `Обучение модели` (Обучить модель на подготовленном датасете, используя размеченные данные; Оценить производительность модели, включая метрики точности и полноты)
   
3) `Оптимизация модели` (Исследовать методы оптимизации модели для улучшения ее производительности; рассмотреть возможности улучшения скорости детекции)

<br>
<br>

<div align="center">
  <h1>Анализ области детекции циклонов</h1>
</div>

В мире существует множество методов и алгоритмов, используемых для решения данной задачи. 

Мы рассмотрели 4 метода:

1) `Сверточные нейронные сети` (CNN). Сверточные нейронные сети являются одним из наиболее популярных методов детекции циклонов. Они обучаются на размеченных изображениях с помеченными циклонами и способны автоматически извлекать признаки, характерные для циклонов, такие как круглые формы и спиральные структуры.

2) `R-CNN` (Region-based Convolutional Neural Network). Метод детекции объектов, который предлагает регионы, где могут находиться объекты, а затем классифицирует их. Он работает следующим образом: сначала извлекает пропозалы (кандидаты на объекты), затем вырезает и масштабирует регионы из изображения и применяет сверточную нейронную сеть для классификации и локализации объектов.

3) `Fast и Faster R-CNN` (Region-based Convolutional Neural Network). Улучшенные версии метода R-CNN, которые работают быстрее.
   
4) `Модели YOLO` (You Only Look At Once). Он примечателен тем, что он способен одновременно определять и классифицировать объекты на изображении. Это означает, что он может обнаруживать наличие циклонов и указывать их местоположение на изображениях. 

<br>
<br>

<div align="center">
  <h1>Формулировка критериев для определения лучшего алгоритма</h1>
</div>

1) **Точность детекции**: Оценка способности каждого алгоритма точно обнаруживать циклоны на изображениях. Алгоритм, обеспечивающий наивысшую точность, будет предпочтительным.
   
2) **Скорость обработки**: Оценка скорости выполнения каждого алгоритма. Быстрый алгоритм может быть более эффективным для оперативного мониторинга циклонов и реагирования на них.

3) **Устойчивость к шуму и изменчивости данных**: Оценка того, насколько хорошо каждый алгоритм справляется с шумом на изображениях и изменениями в условиях наблюдения, такими как разные времена суток и погодные условия.
 
4) **Ресурсоемкость**: Оценка необходимости вычислительных и вычислительных ресурсов для работы каждого алгоритма. Эффективность использования ресурсов может быть важным фактором, особенно при применении в реальном времени.

<br>
<br>

<div align="center">
  <h1>Сравнительный анализ алгоритмов согласно критериям</h1>
</div>

Нужно провести анализ, для того, чтобы выбрать самый подходязий под критерии алгоритм, который поможет решить задачу быстрее и качественнее, при этом потратив минимум ресурсов

<br>
## Сверточные нейронные сети (CNN):

**Плюсы**

1) _CNN_ обычно обеспечивают **высокую точность**
   
2) В зависимости от архитектуры и вычислительных ресурсов, _CNN_ могут быть очень **эффективными**

3) _CNN_ могут быть **устойчивыми** к шуму и изменчивости данных

**Минусы**

1) _CNN_ требует **большого количества данных**

2) _CNN_ очень **вычислительно затратны**

3) _CNN_ могут быть устойчивы к шуму, только если обучены на 

4) _CNN_ требует значительных вычислительных ресурсов

<br>

## R-CNN,  Fast R-CNN и Faster R-CNN

ффф

<br>

## Модели YOLO (You Only Look One Look)

<br>
<br>

<div align="center">
  <h1>Описание алгоритма для решения задачи детекции циклонов</h1>
</div>

Из прошлого пункта видно, что использование готовых моделей YOLO более подходящее решение для данной задачи, плюсов однозначно больше чем минусов. 
Исходя из этого анализа, мы предлагаем алгоритм для детекции циклонов с использованием YOLO.

<br>

## Алгоритм на основе YOLO:

- **Подготовка данных**: Собрать разнообразные и аннотированные данные об изображениях с циклонами и без них. Эти данные будут использоваться для обучения модели (предполагается, что данные уже подготовлены компанией, дающей задание).
  
- **Обучение модели YOLO**: Обучить модель YOLO на подготовленных данных. Модель будет обучаться на изображениях циклонов и учиться выявлять признаки, связанные с циклонами, например спиральность (подразумевается детекция центра циклона).
  
-** Настройка параметров**: Настроить параметры модели YOLO для достижения максимальной точности детекции центра циклонов. Это включает в себя настройку пороговых значений для классификации и локализации объектов.

-**Инференс (предсказание)**: Применить обученную модель YOLO к новым данным. Модель будет анализировать изображения в реальном времени и выделять области, которые считаются центрами циклонов.

- **Постобработка**: Провести постобработку результатов для улучшения точности детекции циклонов. Это может включать в себя фильтрацию ложных срабатываний и объединение близко расположенных областей.

- **Визуализация и оповещение**: Визуализировать результаты детекции циклонов на картах и предоставить операторам или метеорологам информацию о расположении и характеристиках циклонов.

















































