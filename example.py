import automatedcleaning as ac

df = ac.load_data("Titanic.csv")
df = ac.clean_data(df)