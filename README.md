# Heavy Metal Adsorption AI

Файлы:
- `app.py` — Streamlit-приложение
- `Dataset. S.Lamsiah.xlsx` — положите рядом с app.py
- `requirements.txt` — зависимости

Локальный запуск:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Для Streamlit Community Cloud загрузите в один GitHub-репозиторий:
1. app.py
2. Dataset. S.Lamsiah.xlsx
3. requirements.txt

Важно:
- модель прогнозирует Adsorption capacity (mg/g);
- метрики в приложении рассчитываются на случайном train/test split 80/20;
- прогноз является исследовательским, а не лабораторным заключением.
