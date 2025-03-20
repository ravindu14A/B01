import DataFrame_genrator

df = DataFrame_genrator.generate_Dataframe_vector_xyz('data')

#filtered_sorted_df = df[df['Station'] == "BABH"].sort_values(by="Date")



#print(filtered_sorted_df)
df = DataFrame_genrator.convert_to_columns_xyz(df)

df = DataFrame_genrator.convert_geodetic(df)

DataFrame_genrator.save_csv_Dataframe(df, r'geodetic_data_columns_xyz.csv')