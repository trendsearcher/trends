					���������� ������ ����� ��������� �������
_________________________________________________________________________________________________

1) �� ��������� metatrader ����������� ������� ������. ������ ������ ���� ������ ��� �� �������. 
   ������� ������� ���������� �������������� �� � ������� ������� purifier_zipifier_of_row_data

   ���� : 'trends_data/preprocess/SBER616.csv'  
   �����: 'trends_data/preprocess/pureSBER616.csv'

2) ���������� ����� ������� ���� ������������ ����� ��� ������ ���� ��������� ���������� � ������.
   ���� ������� ����� ��������� ������������ O(n) � ����������� �� ����� ������ � �������� ��� �����, 
   ������� ������������� �� ���� ������������. ��� ������ ���� �������� ������� ������ ������ �������� �� 
   2 �����. ������ ���� - ��� ������ ������� ultrasimplefast �� ������ �� 1 ����, � ������� ��������� 
   ��������������-�������������� ����� ����.

   ���� : 'trends_data/preprocess/pureSBER616.csv'  
   �����: 'trends_data/preprocess/shit.csv'

3) ������ ���� ������ ���������� � ������ - ��� ������ ������� broken_trands_postfactum �� ������ 2 ����. 
   ���� ������ ������� ������������� ���� �� ������, ������� ���� ������� �������� ���� � ��������� �������

   ���� 1 :'trends_data/preprocess/pureSBER616.csv'
   ���� 2 :'trends_data/preprocess/shit.csv'
   �����  :'trends_data/preprocess/normal_trends3.csv'

   
4) ���������� ����� ������� ����� ������ ����� ��������� ��������� ���������� ���������� ��������, 
   � ����� ������, ������������� ������ ������ ���� � ����� (����� �������). ������������� ��� ������� 
   ����������� ������������� � ������ �� ������, ���� � ������� ������� �������������, ��������� ��� 
   ������������ � ���������� ��� ��������� ������ � �������� ������������� up-down. ���� ����, 
   � ����� ������������ ���������� ��� � ���������� ������ � ������ ���������� ����������� 
   dubler_remover_after_brokenpostfactumtrends_timeseries. (��������� �� ���������� ����� � ��������������,
   �� ������ �� ����� ������ �� ������������ � ��� ����������, �� ���������������� ������� � � ����� ������)

   ���� 1 :'trends_data/preprocess/pureSBER616.csv'
   ���� 2 :'trends_data/preprocess/normal_trends3.csv'
   �����  :'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors.csv'

5) ����� ����� ���������� ���������������� ������ �������. �� ��� �������� automatic evaluation_surface_timeseries.
   ������ ������ ����� ������������ ����, ��������������� �������������. (��� ��� �� ���� �������� �� ����
   ���������� ��������). ������ = onehot(����� ���������� �������� ��� ������ ������� ���� � ���� � �������
   ��� � ��� ������ ����� � ������ �����)

   ���� 1  : 'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors.csv'
   ���� 2  : 'trends_data/preprocess/pureSBER616.csv'
   ����� 1 : 'trends_data/preprocess/normal_trends_outofdublers_norm_graded.csv'
   ����� 2 : 'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors_graded.pkl'



					��������
___________________________________________________________________________________________________________

6) ����� �� automatic evaluation_surface_timeseries ����� ���� ����������� ��� ������� CNN, DNN, �������� ��������.

   ���� 1 : 'trends_data/preprocess/normal_trends_outofdublers_norm_graded.csv'
   ���� 2 : 'trends_data/preprocess/normal_trends_outofdublers_norm_TPV_vectors_graded.pkl'
