import automatedcleaning as ac

df = ac.load_data("Company_Data.csv")

# You can download the background image from the github
df = ac.clean_data(df, background_image_path="assets/gradient.png")
