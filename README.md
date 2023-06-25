# Covid Articles Project
## Raw Data/Filter 1

The get_clean_and_entity_data method performs several tasks to clean and extract relevant information from the raw dataset:

1. Read the raw dataset from the specified path.
2. Sample a specified number of rows from the dataset using the sample parameter.
3. Save the sampled data as "sample.csv" (raw data).
4. Read the sampled data from "sample.csv" (raw data).
5.  Replace any missing values (NaN) in the dataset with the string "nan".
6.  Rename the columns of the dataset to match the expected column names: "ID", "Text", "Title", "Code", "Url", and "Date".
7.  Iterate over each row in the dataset.
8.  Check if the "Text" column is not equal to "nan" (indicating a valid article text).
9.  Remove cookies from the "Text" column using the remove_cookies method from the TextProcessing class.
10. Check the length of the cleaned text. If it's less than 300,000 characters (larger size may not be fitten into the nlp model, resulting in memory overflow):
    - Create a new DataFrame df_row containing the "ID" value from the current row.
    - Add the "Url", parsed "Text" (processed by the nlp object), "Title", and "Date" columns to df_row.
    - Extract location information using the get_location method from the nlp_extractor object and update df_row.
    - Extract date information using the get_date method from the nlp_extractor object and update df_row.
    - Perform cardinal number checking using the check_cardinal method from the nlp_extractor object and update df_row.
    - Set the "Text" column of df_row to the original cleaned "Text" value from the row.
    - Append df_row to the final dataset.
11. Save the final dataset to the specified "save_path" using the add_to_csv function as Filter_1.csv.

In summary, this code reads a raw dataset, performs cleaning operations on the text, and extracts relevant information such as locations, dates, and numbers. The cleaned and extracted data are then saved as a filtered dataset for further analysis.

## Filter 2

The get_country_mentions_articles method performs the following tasks:

1. Read the dataset from the specified path using pd.read_csv.
2. Replace any missing values (NaN) in the dataset with the string "nan".
3. Rename the columns of the dataset to match the expected column names: "ID", "Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", and "numbers".
4. Iterate over each row in the dataset.
5. Check if the "Text" column is not equal to "nan" (indicating a valid article text).
6. Call the get_geotext_articles method from the locations_extractor object (an instance of the LocationExtractor class) to extract country mentions and other location mentions from the "spacy_loc" and "flair_loc" columns of the current row.
7. Combine the country mentions into a comma-separated string and assign it to the "Mentions" column of the row.
8. Check if the length of the "Mentions" string is greater than 0.
9. If there are country mentions, create a new DataFrame containing only the current row.
10. Use the add_to_csv function to append the new DataFrame to the specified "save_path" CSV file.

In summary, this code reads a filtered dataset ("Filter 1") and filters out articles that do not mention any countries. The filtered data is then saved as a new dataset ("Filter 2") while preserving the "ID" column.

## Filter 3

