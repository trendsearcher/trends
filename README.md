Этот репозиторий служит хранилищем для набора скриптов для предподготовки данных для машинного обучения, 
а также скриптов с самим обучением для проверки некоторых гипотез. Преследуется цель получить классификатор(ы) с  
достаточной точностью (пока она не сильно достаточная) для использования в роли фильтров, которые смогут отсеять 
поток паттернов, формируемых отдельным скриптом, на основе биржевых данных в реальном времени. Весь сопутствующий 
мусор для взаимодействия с терминалом остался от старого недо-робота и я его сюда не пушил. потом 

1.  Models

    папка с моделями ML и описанием всех гипотез
    
    
2.  Preparation_before_learning

    папка с цепочной грязных скриптов, необходимых для преобразования рыночных тиковых данных в датафрейм с паттернами


3.  Working_ml + trends-ML   

    соответственно папки с PCA на автоматически размеченных данных (новый подход) и
    визуализацией некого старья (ручная разметка данных - старый подход)
    
4.  nice_code_for_demonstration

    собственно более-менее нормальный код для демонстрации. 
    
    
5.  ноутбуки

    визуализация того, как распределены примеры по классам на различных фичах, вытащенных из паттернов хардкодингом

                                  Давайте опишем, на чем основана рабочая гипотеза всей работы
================================================================================
Огромное количество терминалов содержит инструменты технического анализа. Вариаций этой ерунды огромная масса. Предполагаем, 
что чем проще и понятнее инструмент, чем он распространеннее и однозначнее, тем к большему количеству людей он применим.
Самым донным инструментом является линия тренда (в коде это pattern) - линия проведенная по касательной к графику по экстремумам. Ее пересечение графиком цены однозначно и понятно.

Второй момент - это инвариантность масштаба в огромном диапазоне, на которой настаивает Мандельброт. Из чего следует, что линии 
любых масштабов должны восприниматься одинаково. 
Третий момент - нелинейность торгового времени, из чего следует, что для однозначности паттерны должны строиться не в координатах 
(цена/реальное время), а в координатах (цена/мера торгового времени). Нелинейность торгового времени связана с скачками активности 
игроков рынка, что пиводит к увеличению количества операций. Каждой операции соответствует тик, поэтому координатами, в которых 
будем работать должна быть пара (цена/номер тика).

На основании этих предположений можно создать набор данных и проверить, есть ли у этого подхода предсказательная сила.

