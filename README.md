# Trends
Этот репозиторий служит хранилищем для набора скриптов для предподготовки данных для машинного обучения, 
а также скриптов с самим обучением для проверки некоторых гипотез. Преследуется цель получить классификатор(ы) с  достаточной 
точностью (пока она не сильно достаточная) для использования в роли фильтров, которые смогут отсеять поток паттернов, формируемых отдельным скриптом,
на основе биржевых данных в реальном времени. Весь сопутствующий мусор для взаимодействия с терминалом остался от старого недо-робота и я его сюда не пушил. потом 

1.  Models

    папка с моделями ML 
    
    
2.  Preparation_before_learning

    папка с цепочной скриптов, необходимых для преобразования рыночных тиковых данных в датафрейм с паттернами


3.  Working_ml + trends-ML   

    соответственно папки с PCA Стаса на автоматически размеченных данных (новый подход) и
    визуализацией некого старья (ручная разметка данных - старый подход)
    
    
4.  ноутбуки

    визуализация того, как распределены примеры по классам на различных фичах, вытащенных из паттернов хардговнокодингом
    
