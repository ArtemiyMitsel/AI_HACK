Источник: MovieLens 100k (grouplens.org) — бинаризован по rating >= 4.

Задача: Бинарная классификация. Понравится ли пользователю фильм (rating >= 4 → target=1).

a. train.csv - обучающая выборка:
    row_id - уникальный идентификатор наблюдения
    user_id - идентификатор пользователя
    item_id - идентификатор фильма
    timestamp - время оценки
    target - целевая переменная (1 если rating >= 4)

b. test.csv - тестовая выборка:
    row_id - уникальный идентификатор наблюдения
    user_id - идентификатор пользователя
    item_id - идентификатор фильма
    timestamp - время оценки
    target - целевая переменная (оставлен для локальной проверки)

c. users.csv - 1:1 lookup по user_id:
    user_id - ключ связи
    age, gender, occupation, zip_code - признаки пользователя

d. items.csv - 1:1 lookup по item_id:
    item_id - ключ связи
    title, release_date - мета-информация
    action, adventure, ... western - one-hot жанры
