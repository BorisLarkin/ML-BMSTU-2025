import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

# Загрузка данных
@st.cache_data
def load_data():
    data = pd.read_csv('../data/obesity_train.csv', sep=",")
    return data

# Предобработка данных с масштабированием
def preprocess_data(data_in):
    data_out = data_in.copy()
    
    # Feature Engineering
    data_out = data_out.assign(
        Metabolic_Risk_Score = (data_out['FAVC'] * 2) + (data_out['FAF'] < 2).astype(int) + (data_out['TUE'] > 2).astype(int) + 0.5,
        Activity_Score = (5 - data_out['FAF']) + (3 - data_out['TUE']) + (5 - data_out['MTRANS']) - 4,
        Diet_Quality = data_out['FCVC'] - data_out['FAVC'],
        Sedentary_Index = (data_out['FAF'] + data_out['TUE']) / 2
    )
    
    # Масштабирование признаков
    scale_cols = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CALC', 'MTRANS', 'CAEC']
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        data_out[new_col_name] = sc1_data[:,i]
    
    # Добавление BMI
    data_out['BMI'] = data_out['Weight'] / (data_out['Height']**2)
    
    return data_out, sc1

# Инициализация моделей
def init_models():
    models = {
        'gradient_boosting_regr': GradientBoostingRegressor(),
        'random_forest_regr': RandomForestRegressor(),
        'gradient_boosting_class': GradientBoostingClassifier(),
        'random_forest_class': RandomForestClassifier(),
        'knn_class': KNeighborsClassifier(),
        'knn_regr': KNeighborsRegressor(),
        'svc': SVC(probability=True),
        'logreg': LogisticRegression(),
        'tree_class': DecisionTreeClassifier(),
        'tree_regr': DecisionTreeRegressor()
    }
    return models

def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

# Сценарий 1: Метод ближайших соседей
def scenario_knn(data):
    st.header('Обучение модели ближайших соседей')
    
    if st.checkbox('Описание метода', key='knn_desc'):
        st.markdown("""
        Фаза предсказания в методе ближайших соседей:
        1. Вычисляем расстояние от искомой точки до всех точек обучающей выборки
        2. Сортируем расстояния по возрастанию
        3. Выбираем K ближайших соседей
        4. Для классификации возвращаем наиболее частый класс
        """)
        st.latex(r'''d(p,q)= \sqrt{ \sum_{i=1}^{n} (p_i-q_i)^2}''')

    cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=5, step=1, key='knn_folds')
    
    data_len = data.shape[0]
    rows_in_one_fold = int(data_len / cv_slider)
    allowed_knn = int(rows_in_one_fold * (cv_slider-1))
    
    st.write(f'Количество строк: {data_len}')
    st.write(f'Максимальное количество соседей: {allowed_knn}')

    cv_knn = st.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1, key='knn_neighbors')
    
    # Используем только масштабированные признаки
    scale_cols = [col for col in data.columns if '_scaled' in col]
    X = data[scale_cols]
    y = data['Obesity']
    
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=cv_knn), 
                           X, y, scoring='accuracy', cv=cv_slider)
    
    st.subheader('Оценка качества модели')
    st.write('Accuracy по фолдам:')
    st.bar_chart(scores)
    st.write(f'Средняя accuracy: {np.mean(scores):.3f}')

# Сценарий 2: Подбор гиперпараметров для KNN
def scenario_knn_tuning(data):
    st.header('Подбор гиперпараметров для KNN')
    
    cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1, key='knn_tune_folds')
    step_slider = st.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1, key='knn_step')
    
    data_len = data.shape[0]
    rows_in_one_fold = int(data_len / cv_slider)
    allowed_knn = int(rows_in_one_fold * (cv_slider-1))
    
    st.write(f'Количество строк: {data_len}')
    st.write(f'Максимальное количество соседей: {allowed_knn}')

    n_range_list = list(range(1, allowed_knn, step_slider))
    n_range = np.array(n_range_list)
    st.write(f'Проверяемые значения K: {n_range}')
    
    scale_cols = [col for col in data.columns if '_scaled' in col]
    X = data[scale_cols]
    y = data['Obesity']
    
    tuned_parameters = [{'n_neighbors': n_range}]
    clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=cv_slider, scoring='roc_auc')
    clf_gs.fit(X, y)
    
    st.subheader('Результаты подбора параметров')
    st.write(f'Лучшее значение K: {clf_gs.best_params_["n_neighbors"]}')
    
    fig = plt.figure(figsize=(7,5))
    plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
    plt.xlabel('Количество соседей')
    plt.ylabel('ROC AUC')
    st.pyplot(fig)

# Сценарий 3: Сравнение нескольких моделей
def scenario_model_comparison(data):
    st.header('Сравнение моделей классификации')
    
    models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
    clas_models = {
        'LogR': LogisticRegression(max_iter=1000), 
        'KNN_5': KNeighborsClassifier(n_neighbors=5),
        'SVC': SVC(probability=True),
        'Tree': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'GB': GradientBoostingClassifier()
    }
    
    models_select = st.multiselect('Выберите модели для сравнения', models_list, default=['RF', 'GB'])
    
    if not models_select:
        st.warning("Выберите хотя бы одну модель")
        return
    
    scale_cols = [col for col in data.columns if '_scaled' in col]
    X = data[scale_cols]
    y = data['Obesity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    roc_auc_list = []
    
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_proba)
        roc_auc_list.append(roc_auc)
        
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        draw_roc_curve(y_test, y_proba, ax[0])
        
        cm = confusion_matrix(y_test, y_pred, normalize='all')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax[1], cmap=plt.cm.Blues)
        fig.suptitle(f'{model_name} (AUC={roc_auc:.2f})')
        st.pyplot(fig)
    
    if len(roc_auc_list) > 0:
        st.subheader('Сравнение ROC AUC')
        comparison_df = pd.DataFrame({'Model': models_select, 'ROC AUC': roc_auc_list})
        st.bar_chart(comparison_df.set_index('Model'))

# Сценарий 4: Предсказание ожирения по признакам
def scenario_prediction(data, models, scaler):
    st.header('Предсказание ожирения по признакам')

    # Создаем полный DataFrame с колонками как в обучающих данных
    # Используем первые строку данных как шаблон, затем очищаем значения
    input_df = data.drop(columns=['Obesity', 'BMI']).iloc[0:1].copy()
    input_df[:] = 0  # Очищаем все значения

    st.subheader('Выберите параметры:')
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Пол', ['Мужской', 'Женский'])
        age = st.slider('Возраст', 10, 80, 30)
        family_history = st.selectbox('Семейная история ожирения', ['Да', 'Нет'])
        faf = st.slider('Физическая активность', 0, 3, 1)
        favc = st.selectbox('Частое употребление высококалорийной пищи', ['Да', 'Нет'])
    
    with col2:
        tue = st.slider('Время за гаджетами', 0, 3, 1)
        fcvc = st.slider('Потребление овощей (раз в день)', 1, 3, 2)
        ncp = st.slider('Основные приемы пищи в день', 1, 4, 3)
        ch2o = st.slider('Потребление воды (литры в день)', 1, 3, 2)
        calc = st.selectbox('Потребление алкоголя', ['Никогда', 'Иногда', 'Часто', 'Всегда'])
    
    # Заполняем основные признаки
    input_df['Gender'] = 1 if gender == 'Мужской' else 0
    input_df['Age'] = age
    input_df['family_history_with_overweight'] = 1 if family_history == 'Да' else 0
    input_df['FAVC'] = 1 if favc == 'Да' else 0
    input_df['FCVC'] = fcvc
    input_df['NCP'] = ncp
    input_df['CH2O'] = ch2o
    input_df['FAF'] = faf
    input_df['TUE'] = tue
    
    # Обработка категориальных признаков
    calc_mapping = {'Никогда': 0, 'Иногда': 1, 'Часто': 2, 'Всегда': 3}
    input_df['CALC'] = calc_mapping[calc]
    
    # Вычисляем производные признаки
    input_df['Metabolic_Risk_Score'] = (input_df['FAVC'] * 2) + (input_df['FAF'] < 2).astype(int) + (input_df['TUE'] > 2).astype(int) + 0.5
    input_df['Activity_Score'] = (5 - input_df['FAF']) + (3 - input_df['TUE']) + (5 - input_df['MTRANS']) - 4
    input_df['Diet_Quality'] = input_df['FCVC'] - input_df['FAVC']
    input_df['Sedentary_Index'] = (input_df['FAF'] + input_df['TUE']) / 2
    

        # Масштабирование признаков
    scale_cols = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CALC', 'MTRANS', 'CAEC']
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(input_df[scale_cols])
    
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        input_df[new_col_name] = sc1_data[:,i]
    
    #for col in scale_cols:
    #    if col in input_df.columns:
            # Для числовых признаков
    #        input_df[f'{col}_scaled'] = scaler.transform(input_df[[col]])[:,0]
    
    # Выбираем только те признаки, которые использовались при обучении модели
    task_clas_cols = [
        'Gender', 'Age', 'family_history_with_overweight',
        'FAVC', 'FCVC_scaled', 'NCP_scaled', 'CAEC_scaled', 'SMOKE',
        'CH2O_scaled', 'SCC', 'FAF_scaled', 'TUE_scaled', 'CALC_scaled',
        'MTRANS_scaled', 'Metabolic_Risk_Score', 'Activity_Score'
    ]
    
    X_pred = input_df[task_clas_cols]
    
    # Выбор модели
    model_type = st.selectbox('Выберите модель для предсказания', 
                            ['Random Forest', 'Gradient Boosting', 'KNN'])
    
    if model_type == 'Random Forest':
        model = models['random_forest_class']
    elif model_type == 'Gradient Boosting':
        model = models['gradient_boosting_class']
    else:
        model = models['knn_class']

    if st.button('Предсказать'):
        try:
            prediction = model.predict(X_pred)
            proba = model.predict_proba(X_pred)
            
            st.subheader('Результат предсказания')
            st.write(f'Предсказанный класс: {prediction[0]}')
            
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(model.classes_, proba[0])
            ax.set_xlabel('Класс ожирения')
            ax.set_ylabel('Вероятность')
            ax.set_title('Вероятности классов')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")
            st.error(f"Используемые признаки: {X_pred.columns.tolist()}")
            st.error(f"Ожидаемые признаки: {model.feature_names_in_}")

# Обучение моделей
def train_models(models, X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train[name], y_train[name])
        trained_models[name] = model
    return trained_models

# Основное приложение
def main():
    st.set_page_config(layout="wide", page_title="Анализ ожирения")
    
    # Загрузка данных
    st.title("Анализ и прогнозирование ожирения")
    data_load_state = st.text('Загрузка данных...')
    raw_data = load_data()
    data, scaler = preprocess_data(raw_data)
    data_load_state.text('')
    
    # Создание вкладок
    tabs = ["Обзор данных", "Градиентный бустинг (регрессия)", "Градиентный бустинг (классификация)",
            "Случайный лес (регрессия)", "Случайный лес (классификация)", 
            "Метод ближайших соседей", "Подбор параметров KNN", "Сравнение моделей",
         "Предсказание ожирения", "Рекомендации"]
    selected_tab = st.sidebar.radio("Выберите раздел:", tabs)
    
    # Определение признаков для моделей
    # Для классификации
    task_clas_cols = [
        'Gender', 'Age', 'family_history_with_overweight',
        'FAVC', 'FCVC_scaled', 'NCP_scaled', 'CAEC_scaled', 'SMOKE',
        'CH2O_scaled', 'SCC', 'FAF_scaled', 'TUE_scaled', 'CALC_scaled',
        'MTRANS_scaled', 'Metabolic_Risk_Score', 'Activity_Score'
    ]
    
    # Для регрессии
    task_regr_cols = [
        'Gender', 'Age', 'family_history_with_overweight',
        'FAVC', 'FCVC_scaled', 'NCP_scaled', 'CAEC_scaled', 'SMOKE',
        'CH2O_scaled', 'SCC', 'FAF_scaled', 'TUE_scaled', 'CALC_scaled',
        'MTRANS_scaled', 'Metabolic_Risk_Score', 'Activity_Score',
        'Diet_Quality', 'Sedentary_Index'
    ]
    
    # Разделение данных
    X_clas = data[task_clas_cols]
    X_regr = data[task_regr_cols]
    
    # Для регрессии (BMI)
    y_regr = data['BMI']
    
    # Для классификации
    y_class = data['Obesity']
    
    X_train_clas, X_test_clas, y_train_class, y_test_class = train_test_split(
        X_clas, y_class, test_size=0.2, random_state=42)
    
    X_train_regr, X_test_regr, y_train_regr, y_test_regr = train_test_split(
        X_regr, y_regr, test_size=0.2, random_state=42)
    
    # Подготовка данных для обучения
    y_train = {
        'gradient_boosting_regr': y_train_regr,
        'random_forest_regr': y_train_regr,
        'gradient_boosting_class': y_train_class,
        'random_forest_class': y_train_class,
        'knn_class': y_train_class,
        'knn_regr':y_train_regr,
        'svc': y_train_class,
        'logreg': y_train_class,
        'tree_class': y_train_class,
        'tree_regr': y_train_regr
    }
    X_train_dict = {
        'gradient_boosting_regr': X_train_regr,
        'random_forest_regr': X_train_regr,
        'gradient_boosting_class': X_train_clas,
        'random_forest_class': X_train_clas,
        'knn_class': X_train_clas,
        'knn_regr':X_train_regr,
        'svc': X_train_clas,
        'logreg': X_train_clas,
        'tree_class': X_train_clas,
        'tree_regr': X_train_regr
    }
    
    # Инициализация и обучение моделей
    models = init_models()
    trained_models = train_models(models, 
                                X_train_dict, 
                                y_train)
    
    # Вкладка "Обзор данных"
    if selected_tab == "Обзор данных":
        st.header("Обзор данных")
        
        st.subheader("Первые 5 строк данных")
        st.write(data.head())
        
        st.subheader("Статистика данных")
        st.write(data.describe())
        
        st.subheader("Распределение целевых переменных")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Распределение BMI (регрессия)")
            fig, ax = plt.subplots()
            sns.histplot(y_regr, kde=True, ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("Распределение классов ожирения (классификация)")
            fig, ax = plt.subplots()
            sns.countplot(x=y_class, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Корреляционная матрица (масштабированные данные)")
        corr_cols = [col for col in data.columns if '_scaled' in col] + ['BMI', 'Obesity']
        corr_matrix = data[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    # Вкладки для моделей
    elif selected_tab.startswith("Градиентный бустинг") or selected_tab.startswith("Случайный лес"):
        model_type = "gradient_boosting" if "Градиентный" in selected_tab else "random_forest"
        task_type = "regr" if "регрессия" in selected_tab else "class"
        model_name = f"{model_type}_{task_type}"
        model = trained_models[model_name]
        
        st.header(selected_tab)
        
        if task_type == "regr":
            # Регрессия
            X_test = X_test_regr
            y_test = y_test_regr
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.subheader("Метрики качества")
            st.write(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
            st.write(f"Коэффициент детерминации (R²): {r2:.2f}")
            
            st.subheader("График предсказаний vs реальных значений")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.3)
            ax.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Реальные значения BMI')
            ax.set_ylabel('Предсказанные значения BMI')
            st.pyplot(fig)
            
        else:
            # Классификация
            X_test = X_test_clas
            y_test = y_test_class
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            st.subheader("Метрики качества")
            st.write(f"Точность (Accuracy): {accuracy:.2f}")
            st.write(f"F1-мера: {f1:.2f}")
            
            st.subheader("Матрица ошибок")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                         display_labels=model.classes_)
            disp.plot(ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Важность признаков")
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(np.array(X_test.columns)[sorted_idx])
        ax.set_xlabel('Важность признака')
        st.pyplot(fig)
        
        st.write("Топ-5 важных признаков:")
        for i in sorted_idx[-5:][::-1]:
            st.write(f"- {X_test.columns[i]}: {feature_importance[i]:.4f}")
    
    # Вкладка "Сравнение моделей"
    elif selected_tab == "Метод ближайших соседей":
        scenario_knn(data)
    
    elif selected_tab == "Подбор параметров KNN":
        scenario_knn_tuning(data)
    
    elif selected_tab == "Сравнение моделей":
        scenario_model_comparison(data)
    
    elif selected_tab == "Предсказание ожирения":
        scenario_prediction(data, models, scaler)
    
    # Вкладка "Рекомендации"
    elif selected_tab == "Рекомендации":
        st.header("Результаты исследования и рекомендации")
        
        st.subheader("Ключевые выводы")
        st.markdown("""
        1. **Генетическая предрасположенность** (семейная история ожирения) - самый значимый фактор
        2. **Пищевые привычки**:
           - Низкое потребление овощей
           - Частые перекусы между приемами пищи
        3. **Образ жизни**:
           - Малоподвижность (время за гаджетами)
           - Использование транспорта вместо ходьбы
        4. **Возраст** - после 30 лет риск ожирения увеличивается
        """)
        
        st.subheader("Рекомендации по профилактике ожирения")
        st.markdown("""
        - 🥦 **Увеличьте потребление овощей** - минимум 3 порции в день
        - ⏱️ **Контролируйте перекусы** - замените сладости на фрукты/орехи
        - 🚶 **Больше двигайтесь** - 10,000 шагов в день или 150 мин кардио в неделю
        - 🚗 **Реже пользуйтесь транспортом** - ходите пешком на короткие расстояния
        - 🏋️ **После 30 лет** - добавьте силовые тренировки 2-3 раза в неделю
        - 📊 **Регулярно измеряйте BMI** - контроль веса помогает вовремя принять меры
        """)
        
        st.subheader("Персонализированные рекомендации")
        st.write("Введите свои параметры для получения индивидуальных рекомендаций:")
        
        with st.form("recommendation_form"):
            age = st.slider("Возраст", 10, 80, 30)
            family_history = st.selectbox("Семейная история ожирения", ["Да", "Нет"])
            activity = st.selectbox("Уровень активности", 
                                  ["Сидячий", "Умеренный", "Активный"])
            veg_consumption = st.slider("Потребление овощей (раз в день)", 1, 5, 3)
            
            submitted = st.form_submit_button("Получить рекомендации")
            
            if submitted:
                st.success("Персонализированные рекомендации:")
                
                recommendations = []
                if family_history == "Да":
                    recommendations.append("🔹 У вас есть генетическая предрасположенность - особенно важно следить за питанием")
                
                if age > 30:
                    recommendations.append(f"🔹 В вашем возрасте ({age} лет) метаболизм замедляется - уменьшите калорийность рациона на 5-10%")
                
                if activity == "Сидячий":
                    recommendations.append("🔹 У вас сидячий образ жизни - старайтесь делать перерывы каждые 30 минут для 5-минутной активности")
                
                if veg_consumption < 3:
                    recommendations.append(f"🔹 Вы едите мало овощей ({veg_consumption} раза в день) - увеличьте до 3-5 порций")
                
                if not recommendations:
                    st.write("Ваши привычки выглядят здоровыми! Продолжайте в том же духе.")
                else:
                    for rec in recommendations:
                        st.write(rec)

if __name__ == "__main__":
    main()