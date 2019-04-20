					ПОДГОТОВКА ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ МОДЕЛЕЙ
_________________________________________________________________________________________________

1) из терминала metatrader скачиваются тиковые данные. формат записи этих данных нам не годится. 
   поэтому сначала необходимо предобработать их с помощью скрипта purifier_zipifier_of_row_data

   вход : 'trends_data/preprocess/SBER616.csv'  
   выход: 'trends_data/preprocess/pureSBER616.csv'

2) полученный таком образом файл используется затем для поиска всех возможных кандидатов в тренды.
   этот процесс имеет нелинейно возрастающую O(n) в зависимости от длины тренда и масштаба той сетки, 
   которая накладывается на окно сканирования. Для обхода этой проблемы процесс поиска тренод разделен на 
   2 этапа. первый этам - это запуск скрипта ultrasimplefast на данных из 1 шага, в котором применена 
   гиперболически-распределенная сетка окна.

   вход : 'trends_data/preprocess/pureSBER616.csv'  
   выход: 'trends_data/preprocess/shit.csv'

3) второй этап поиска кандидатов в тренды - это запуск скрипта broken_trands_postfactum на данных 2 шага. 
   этот скрипт всецело характеризует лишь те тренды, которые были пробиты графиком цены в ближайшем будущем

   вход 1 :'trends_data/preprocess/pureSBER616.csv'
   вход 2 :'trends_data/preprocess/shit.csv'
   выход  :'trends_data/preprocess/normal_trends3.csv'

   
4) полученный таким образом набор данных может содержать небольшое количество идентичных дублеров, 
   а также тренды, расположенные крайне близко друг к другу (почти дублеры). исключительно для чистоты 
   последующих экспериментов я удаляю те тренды, окна в будущее которых перекрываются, поскольку они 
   используются в дальнейшем для вынесения оценки в бинарной классификации up-down. этот этап, 
   а также выравнивание количества фич и приведение записи в формат датафрейма осуществяет 
   dubler_remover_after_brokenpostfactumtrends_timeseries. (некоторые из полученных фичей я закоментировал,
   не смотря на явную пользу от содержащейся в них информации, их закоментирование привело в к росту метрик)

   вход 1 :'trends_data/preprocess/pureSBER616.csv'
   вход 2 :'trends_data/preprocess/normal_trends3.csv'
   выход  :'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors.csv'

5) после этого необходимо охарактеризовать тренды оценкой. за это отвечает automatic evaluation_surface_timeseries.
   данный скрипт также рассчитывает фичу, характеризующую волатильность. (мне она не дала прироста на моем
   игрушечном датасете). оценка = onehot(сумма интегралов площадей под крифой графика цены в окне в будущее
   под и над точкой входа с учетом знака)

   вход 1  : 'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors.csv'
   вход 2  : 'trends_data/preprocess/pureSBER616.csv'
   выход 1 : 'trends_data/preprocess/normal_trends_outofdublers_norm_graded.csv'
   выход 2 : 'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors_graded.pkl'



					ОБУЧЕНИЕ
___________________________________________________________________________________________________________

6) выход из automatic evaluation_surface_timeseries может быть использован для моделей CNN, DNN, ансамбля нейронок.

   вход 1 : 'trends_data/preprocess/normal_trends_outofdublers_norm_graded.csv'
   вход 2 : 'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors_graded.pkl'
