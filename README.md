
Данный репозиторий создан в рамках прохождения курса **"MLOps и production подход к ML исследованиям"** и реализации финального проекта **"MLOps на примере задачи матчинга геолокаций (на основе соревнования Kaggle Foursquare - Location Matching)"**

**Ссылки на курс:**
- https://ods.ai/tracks/ml-in-production-spring-22
- https://yandex.ru/q/article/osnovnaia_informatsiia_dlia_uchastnikov_418642d4/


### Полный дневник прохождения курса и реализации финального проекта, в котором подробно и пошагово расписаны все этапы работы над домашними заданиями и процесс разработки финального проекта:
	- PDF вариант ./Docs/Notes_for_Cource_MLOps_and_production_ML.pdf
	- HTML вариант ./Docs/Notes_for_Cource_MLOps_and_production_ML.html

### Краткое описание проекта и презентация:
	- ./Docs/MLOps project Location Matching (description).pdf
	- ./Docs/MLOps project Location Matching (presentation).pdf



### Инструкция по запуску всей инфраструктуры финального проекта на новой машине (инструкция под Windows):

- Примечания: ниже описаны шаги от самого начала и до запуска целевого микросервиса, при этом, для запуска целевого микросервиса, необходимо будет связать локальный репозиторий со своим удаленным, т.к. доставка и деплой сервиса осуществляется через гитлаб пайплайн, и какой-либо комит в main ветку, для нижеуказанного репозитория права есть только у участников команды mlops-23reg-team
- Устанавливаем предварительно в систему:
	- python 3.9
	- poetry
	- Docker Desktop (в виде бэкенда WSL2)
- git clone https://gitlab.com/mlops-23reg-team/mlops-23reg-project.git
- все нижеуказанные команды выполняем находясь в корневой директории проекта
- создаем в корне проекта файл mlops-23reg-team следующего содержания:
```
MINIO_ROOT_USER = admin
MINIO_ROOT_PASSWORD = miniotestpass
AWS_ACCESS_KEY_ID = L6Vt9Pw72XDw26Mt
AWS_SECRET_ACCESS_KEY = IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
AWS_S3_MLFLOW_BUCKET = mlflow
AWS_S3_DVC_BUCKET = dvc
MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
MLFLOW_TRACKING_URI = "http://localhost:5000"
POSTGRES_USER = dbuser
POSTGRES_PASSWORD = dbtestpass
POSTGRES_DB = mlflow
PGADMIN_DEFAULT_EMAIL = "admin@admin.com"
PGADMIN_DEFAULT_PASSWORD = pgtestpass
```
- создаем файл .dvc/config.local следующего содержания
```
['remote "s3minio"']
    access_key_id = L6Vt9Pw72XDw26Mt
    secret_access_key = IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
```
- docker-compose up -d --build (дожидаемся старта контейнеров)
- производим предварительную настройку minio s3
	- Заходим через браузер http://127.0.0.1:9001 и авторизуемся admin/miniotestpass
	- Через веб-интерфейс создаем два бакета:
		- Buckets -> Create bucket
			- mlflow
			- dvc
	- Через веб-интерфейс создаем сервисный аккаунт (по сути API Key):
		- Identity -> Service Accounts -> Create service account
		- Name: L6Vt9Pw72XDw26Mt
		- Pass: IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
- Добавляем в проект исходные данные (датасеты)
	- Заходим в соревнование https://www.kaggle.com/competitions/foursquare-location-matching/data 
	- копируем из этого соревнования файлы train.csv, test.csv, sample_submission.csv, pairs.csv в директорию проекта ./data/raw/
- Создаем виртуальную среду
	- poetry install
	- poetry shell
- Запускаем эксперимент
	- dvc repro
- Заходим в mlflow через http://127.0.0.1:5000 
	- проверяем что трекинг эксперимента успешно добавлен
	- заходим в модели и выставляем для текущей модели stage = staging
- Все последующие шаги возможны, только после привязки локального репозитория к удаленному в гитлабе, в который есть полный доступ (перепривязываем в какой-либо свой)
- Для запуска целевого сервиса, на текущей машине необходимо поднять гитлаб раннер (подробная инструкция есть в подразделе "Настраиваем локальный CI Runner" дневника курса)
- В удаленном репозитории gitlab в Settings -> CI/CD -> Variables создаем следующие переменные:
	- AWS_ACCESS_KEY_ID (Protected: False, Masked: True): L6Vt9Pw72XDw26Mt
	- AWS_SECRET_ACCESS_KEY (Protected: False, Masked: True): IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
	- MLFLOW_S3_ENDPOINT_URL (Protected: False, Masked: False):  http://<ip_docker_server_host>:9000 (<ip_docker_server_host> узнаем через PowerShell выполнив wsl hostname -I (должна быть именно заглавная английская "и")
	- MLFLOW_TRACKING_URI (Protected: False, Masked: False):  http://<ip_docker_server_host>:5000
	- APP_PROJECT_VERSION(Protected: False, Masked: False): 0.1.0
- git tag -af v0.1.0 -m "Project version 0.1.0"
- git push v0.1.0 (или git push --tags, в этом случае запушатся все локальные теги)
- git push origin main
- По итогу отработки всех gitlab пайплайнов должны получить работающий контейнер dev_ml_service и API по адресу http://127.0.0.1:8004/invocations:
- Проверить работу API можно через postman
	- POST
	- http://127.0.0.1:8004/invocations
	- BODY -> from-data
	- в первой строке, в поле key вводим "file" (это имя параметра функции async def create_upload_file(file: UploadFile = File(...)), в этой же ячейке справа в выпадающем списке выбираем File
	- указываем предварительно подготовленный файл .csv с несколькими строками, исходного формата датафрейма, например такого содержания (важно, что для текущего варианта бейзлайна, как минимум 2 POI должны находиться географически рядом, ближе чем на 100м друг к другу, иначе алгоритм не сможет сгенерировать парный файл, это легко дорабатывается в коде, но в текущем проекте не было задачи править подобное поведение):
```
id,name,latitude,longitude,address,city,state,zip,country,url,phone,categories
E_00001118ad0191,Jamu Petani Bagan Serai,5.012169,100.535805,,,,,MY,,,Cafés
E_000020eb6fed40,Johnny's Bar,40.43420889065938,-80.56416034698486,497 N 12th St,Weirton,WV,26062,US,,,Bars
E_00002f98667edf,QIWI,47.215134,39.686088,"Межевая улица, 60",Ростов-на-Дону,,,RU,https://qiwi.com,+78003011131,ATMs
E_001b6bad66eb98,"Gelora Sriwijaya, Jaka Baring Sport City",-3.01467472168758,104.79437444575598,,,,,ID,,,Stadiums
E_0283d9f61e569d,Stadion Gelora Sriwijaya,-3.021726757527373,104.78862762451172,Jalan Gubernur Hasan Bastari,Palembang,South Sumatra,11480.0,ID,,,Soccer Stadiums
E_00002f98667edf_copy,QIWI,47.215134,39.686088,"Межевая улица, 60",Ростов-на-Дону,,,RU,https://qiwi.com,+78003011131,ATMs
E_001b6bad66eb98_copy,"Gelora Sriwijaya, Jaka Baring Sport City",-3.01467472168758,104.79437444575598,,,,,ID,,,Stadiums 
```
	- нажимаем send, по итогу должны получить ответ в виде отданного .csv файла, для примера выше, ответ будет следующим:
```
id,matches
E_00001118ad0191,E_00001118ad0191
E_000020eb6fed40,E_000020eb6fed40
E_00002f98667edf,E_00002f98667edf E_00002f98667edf_copy
E_001b6bad66eb98,E_001b6bad66eb98 E_001b6bad66eb98_copy
E_0283d9f61e569d,E_0283d9f61e569d
E_00002f98667edf_copy,E_00002f98667edf_copy E_00002f98667edf
E_001b6bad66eb98_copy,E_001b6bad66eb98_copy E_001b6bad66eb98
```
