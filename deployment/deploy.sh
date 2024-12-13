#!/bin/bash

# Путь к папке с вашим приложением
APP_DIR="/path/to/your/app"

# Путь к репозиторию
REPO_URL="git@github.com:username/repo.git"

# Имя ветки, которую нужно развернуть
BRANCH="main"

# Команда для запуска вашего сервера (например, uvicorn для FastAPI)
START_CMD="uvicorn main:app --host 0.0.0.0 --port 8000"

# Функция для логирования ошибок
log_error() {
  echo "[ERROR]: $1" >&2
}

# Переход в директорию приложения
cd "$APP_DIR" || { log_error "Не удается перейти в папку $APP_DIR"; exit 1; }

# Fetch последней версии кода
git fetch origin || { log_error "Не удается получить обновления из репозитория"; exit 1; }

# Проверка на наличие изменений
if git diff-index --quiet HEAD --; then
  echo "Код в актуальном состоянии, ничего не меняем"
else
  # Старайтесь сохранить пользовательские изменения (например, с помощью git stash)
  git stash || { log_error "Не удается сохранить текущие изменения"; exit 1; }

  # Switch to the desired branch и pull обновления
  git checkout $BRANCH || { log_error "Не удается перейти на ветку $BRANCH"; exit 1; }
  git pull origin $BRANCH || { log_error "Не удается обновить код из ветки $BRANCH"; exit 1; }

  # Восстановить зависимость (например, для Python)
  pip install -r requirements.txt || { log_error "Не удается установить зависимости"; exit 1; }

  # Запуск или перезапуск сервиса
  pkill -f "uvicorn" || true  # Не обязательно, если вы хотите просто завершить старый процесс
  nohup $START_CMD > app.log 2>&1 &

  # Восстановление изменений в рабочем дереве, если нужно
  git stash pop || { log_error "Не удается восстановить изменения в рабочем дереве"; exit 1; }
fi
