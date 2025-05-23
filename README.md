## 19. Система анализа задач службы технической поддержки с авторекомендациями.

**Куратор** - Руслан Каюмов @KayumovRu

**Участники**:
- Ермаков Павел Алексеевич @Patapon
- Ефимова Елена Сергеевна @ElenaSergeevna
- Рудаш Кирилл Андреевич @KirillKirilych
- Филоненко Феодосия Юрьевна @andiefreude
- Ильин Илья Владимирович @ilyinily

Проект включает в себя прогнозирование временных рядов и NLP , направлен на разработку системы, которая будет анализировать данные о прошлых обращениях в службу поддержки и предоставлять прогнозы о будущих обращениях.

Прогнозирование временных рядов позволит предсказывать количество и характер обращений клиентов в будущем, что поможет лучше планировать ресурсы и предотвращать возможные проблемы. Это может включать в себя анализ трендов и других факторов, влияющих на частоту и тип обращений.

NLP же позволит системе понимать и обрабатывать текстовые данные, такие как описания проблем и запросы пользователей. С помощью NLP можно будет автоматически классифицировать обращения, извлекать ключевые слова и фразы, а также определять тональность и контекст сообщений.

Наш проект разделен на 2 части (TS, NLP), и для 2 частей мы используем разные датасеты.

## Особенности TS

- Нагрузка на очереди имеет иерархическую структуру. Одна очередь может иерархически иметь глубину - 6 очередей. И ожидается, что клиента будет интересовать нагрузка на конкретную очередь, но при этом мы должны включать все дочерние очереди.

## Особенности NLP

- В датасете есть много сокращений и слов на английском языке, которые были предобрабатывались для кластеризации. В стоп-слова добавлены слова, которые ухудщали coherence score.

## Файлы с выводами

- [checkpoints, описание чекпоинтов](./docs/checkpoints.md)
- [dataset, описание данных](./docs/dataset.md)
- [EDA, анализ данных](./docs/EDA.md)
- [baseline, Описание базовых моделей](./docs/BASELINE.md)
- [checkpont5_results, Результаты пятого чекпоинта](./docs/checkpoint5_results.pdf)
- [checkpoint6_results, Результаты шестого чекпоинта](./docs/checkpoint6_results.md)
- [Документация проекта](./docs/report.pdf)

## Сервис

### Frontend

Сервис написан на фреймворке Streamlit. В нем есть две страницы - для NLP и TS.

###  Backend

Вебсервис реализован на веб-фреймворке FastAPI.

### Запуск

Так как фронтенд и бэкенд почти всегда запускаются вместе, то для локального запуска используется `docker compose up backend frontend -d --build` или `just service-build`
Фронтенд будет находиться на локальном порту `http://0.0.0.0:8501/`

## Инфраструктура

### Перед запуском

- Чтобы скачать данные, надо указать логин, пароль от инстанса MiniO в `.env` (в беседе)
- Установить `poetry`
- Запустить `poetry install`
- Установить `just` для удобства

### Структура кода

- `data/` - данные проекта, подгружаются отдельной командой
- `notebooks/` - ноутбуки с исследованием данных, качества моделей (TS + NLP)
- `scripts/` - полезные скрипты
- `tasks_support_system_ai/` - пакет со всем функционалом, который будет использоваться для работы сервиса
- `tests/` - тесты пакета
- `nginx/` - конфиг для `nginx` для miniO и самого сервиса
- `docs/` - документы, шаблон PR

### Данные

- Данные лежат в инстансе MiniO, куда можно пушить и откуда пуллить данные через `just pull-data` и `just push-data`.
- Инстанс поднимается через `docker compose`, код которого находится в данном репозитории.

### Контроль качества кода

- Настроена автопроверка тестов, правил линтера и форматтера.
- В качестве линтера и форматтера используется `ruff`. Используется большое кол-во правил.
- Для автоисправления локально можно использовать рецепт `just full-style`.

### Особенности

- Используем `just` для запуска частых команд. Он похож на `make`, но удобней из-за специализации не на сборке, а на запуске команд.
- Зависимости устанавливаем через `poetry`
- Папка `./tasks_support_system_ai/` - это python пакет, который устанавливается в окружение через `poetry`. Это помогает значительно облегчить жизнь с импортами модулей внутри пакета.

### Демонстрация работы сервиса

- Fast API

![image](https://github.com/user-attachments/assets/44dda48b-44a4-4b8f-99ad-e8a5002904bf)

- Streamlit - приложение

![image](https://github.com/user-attachments/assets/864bef1b-6d9b-4456-80c2-4c82dd72bdb7)

### Разработка

- Установить `poetry`
- Установить `just`
- Установить `docker`/`podman`
- Установить `.env` файл нужными значениями (в закрепе чата)
- Получить данные `just pull-data` или у товарища p2p
- Придерживаемся [trunk-based development](https://trunkbaseddevelopment.com/). Ветки короткоживущие, не делаем merge коммитов, только rebase!

#### Однокнопочная команда для проверки установки и настройки окружения:
- `just setup` для установки конфигов, проверки `.env` файла, установки зависимостей, скачивания данных

#### Данные

Данные находятся в DVC хранилище (на основе того же minio).
Пользоваться также:
- `just pull-data` или `dvc pull`, чтобы скачать данные
- `dvc commit -m 'Commit message to data'` и `just push-data` или `dvc push` для обновления данных

В конфиге dvc стоит autostaging, но если он не заработал, то при обновлении данных надо писать `dvc add data/nlp` к примеру.

#### Trunk-Based Development (TBD)

Пользоваться самописными функциями на свой риск, но приветствуется PR по этому поводу.

Предпосылки:
- remote называется `origin`
- **Локальные ветки сразу пушим в удаленный репозиторий**, иначе есть риск потери изменений
- Удаленные ветки автоматически удаляются при мердже PR

Для упрощения TBD, есть команды для упрощения работы (могут ломаться на разных OS, нужен фидбек):

- `just list-branches` - посмотреть локальные ветки
- `just clean-branches` - удалить ветки, которых нет в удаленном репозитории.
- `just recreate-branch` - если была смерджена ветка, но нравится её название, то эта ветка будет отведена от актуального состояния main, чтобы не делать ручных действий.
- `just update-main` - обновить main (не знаю, насколько полезно).
