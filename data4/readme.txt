Задача: бинарная классификация. Нужно предсказать вероятность кредитного дефолта клиента Home Credit.

Основная таблица:
- train.csv - обучающая выборка по клиентам.
- test.csv - тестовая выборка по клиентам.
- SK_ID_CURR - идентификатор клиента / заявки. Это ключ для объединения таблиц.
- target - целевая переменная в train.csv. 1 = дефолт, 0 = нет дефолта.

Правило объединения:
- Все вспомогательные таблицы, где есть SK_ID_CURR, нужно присоединять к train.csv и test.csv по ключу SK_ID_CURR.
- Использовать left join по SK_ID_CURR.
- Если после объединения есть пропуски, это означает отсутствие соответствующей истории или рассчитанного score.

Вспомогательные таблицы со score-признаками:
- train_doc_score.csv, test_doc_score.csv - score по документным признакам. Полезная числовая колонка: doc_score.
- train_house_score.csv, test_house_score.csv - score по жилищным признакам. Полезная числовая колонка: house_score.
- pos_score_train.csv, pos_score_test.csv - score из истории POS/CASH. Полезная числовая колонка: pos_score.
- inst_score_train.csv, inst_score_test.csv - score из истории installments payments. Полезная числовая колонка: inst_score.
- cc_score_train.csv, cc_score_test.csv - score из истории credit card balance. Полезная числовая колонка: cc_score.
- bubl_score_train.csv, bubl_score_test.csv - score из bureau/bureau_balance. Полезная числовая колонка: bubl_score.

Вспомогательные таблицы с предсказаниями отдельных моделей:
- train_pred_lgb1.csv, train_pred_lgb2.csv, train_pred_lgb3.csv - out-of-fold предсказания отдельных LightGBM моделей для train. Полезная числовая колонка: prob.
- В train_pred_lgb*.csv колонка target является служебной копией целевой переменной из train и не должна использоваться как признак.
- test_pred_lgb1.csv, test_pred_lgb2.csv, test_pred_lgb3.csv - предсказания тех же моделей для test. Колонка TARGET в этих файлах содержит score модели, а не истинный таргет.

Прочие файлы:
- house_ex.csv - дополнительная таблица с house_score по SK_ID_CURR.
- lgb1_feature_importance.csv - служебная таблица важностей признаков без ключа SK_ID_CURR, для генерации признаков не использовать.
- stacked_sub.csv - служебный файл с итоговыми тестовыми предсказаниями без train-аналога, для генерации признаков не использовать.

Рекомендация для генерации признаков:
- Самые естественные признаки для этой задачи - прямое присоединение score-колонок по SK_ID_CURR и арифметические комбинации нескольких score между собой.
