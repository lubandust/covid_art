# Covid Articles Project
## Raw Data/Filter 1

The `get_clean_and_entity_data` method performs several tasks to clean and extract relevant information from the raw dataset:

1. Read the raw dataset from the specified path.
2. Sample a specified number of rows from the dataset using the `sample` parameter.
3. Save the sampled data as "sample.csv".
4. Read the sampled data from "sample.csv".
5. Replace any missing values (NaN) in the dataset with the string "nan".
6. Rename the columns of the dataset to match the expected column names: "ID", "Text", "Title", "Code", "Url", and "Date".
7. Iterate over each row in the dataset.
8. Check if the "Text" column is not equal to "nan" (indicating a valid article text).
9. Remove cookies from the "Text" column using the `remove_cookies` method from the `TextProcessing` class.
10. Check the length of the cleaned text. If it's less than 300,000 characters:
    - Create a new DataFrame `df_row` containing the "ID" value from the current row.
    - Add the "Url", parsed "Text" (processed by the `nlp` object), "Title", and "Date" columns to `df_row`.
    - Extract location information using the `get_location` method from the `nlp_extractor` object and update `df_row` (both for flair and spacy methods).
    - Extract date information using the `get_date` method from the `nlp_extractor` object and update `df_row`.
    - Perform cardinal number checking using the `check_cardinal` method from the `nlp_extractor` object and update `df_row`.
    - Set the "Text" column of `df_row` to the original cleaned "Text" value from the row.
    - Append `df_row` to the final dataset.
11. Save the final dataset to the specified "save_path" using the `add_to_csv` function.

In summary, this code reads a raw dataset, performs cleaning operations on the text, and extracts relevant information such as locations, dates, and numbers. The cleaned and extracted data are then saved as a filtered dataset for further analysis.

## Filter 2

The `get_country_mentions_articles` method performs the following tasks:

1. Read the dataset from the specified path using `pd.read_csv`.
2. Replace any missing values (NaN) in the dataset with the string "nan".
3. Rename the columns of the dataset to match the expected column names: "ID", "Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", and "numbers".
4. Iterate over each row in the dataset.
5. Check if the "Text" column is not equal to "nan" (indicating a valid article text).
6. Call the `get_geotext_articles` method from the `locations_extractor` object (an instance of the `LocationExtractor` class) to extract country mentions and other location mentions from the "spacy_loc" and "flair_loc" columns of the current row.
7. Combine the country mentions into a comma-separated string and assign it to the "Mentions" column of the row.
8. Check if the length of the "Mentions" string is greater than 0.
9. If there are country mentions, create a new DataFrame containing only the current row.
10. Use the `add_to_csv` function to append the new DataFrame to the specified "save_path" CSV file.

In summary, this code reads a filtered dataset ("Filter 1") and filters out articles that do not mention any countries. The filtered data is then saved as a new dataset ("Filter 2") while preserving the "ID" column.

## Filter 3

The `get_us_articles` method performs the following tasks:

1. Read the dataset from the specified path using `pd.read_csv`.
2. Replace any missing values (NaN) in the dataset with the string "nan".
3. Rename the columns of the dataset to match the expected column names: "ID", "Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", "numbers", and "Mentions".
4. Iterate over each row in the dataset.
5. Check if the "Text" column is not equal to "nan" (indicating a valid article text).
6. Call the `get_US_geotext_articles` method from the `locations_extractor` object (an instance of the `LocationExtractor` class) to extract mentions related to the United States (US) from the "Mentions" column of the current row.
7. If the article is related to the US, retrieve the US city mentions and assign them to the "US_cities" column of the row.
8. Create a new DataFrame containing only the current row.
9. Use the `add_to_csv` function to append the new DataFrame to the specified "save_path" CSV file.

In summary, this code reads a filtered dataset ("Filter 2") and further narrows it down to articles specifically related to the United States (US). The filtered data is then saved as a new dataset ("Filter 3") while preserving the "ID" column.

## Filter 4

The `apply_strategies` method performs the following tasks:

1. Read the dataset from the specified path using `pd.read_csv`.
2. Read the `uscities.csv` file as `df_simplemaps` using `pd.read_csv`. This file contains information about US cities.
3. Replace any missing values (NaN) in the dataset with empty strings.
4. Rename the columns of the dataset to match the expected column names: "ID", "Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", "numbers", "Mentions", and "US_cities".
5. Create a list of unique state names from the `df_simplemaps` dataset.
6. Iterate over each row in the dataset.
7. Convert the "US_cities" column from a string representation to a list.
8. Initialize empty fields for "counties", "states", "strategy_1", and "strategy_2" in the current row.
9. For each city in the "US_cities" list:
   - If the city is a state name, add it to the "states" field in the current row.
   - If the city is not a state name, query the `df_simplemaps` dataset to retrieve the corresponding counties and states for the city.
   - Append the county names and state names to the "counties" and "states" fields in the current row.
10. If the "counties" field is not empty, call the `get_strategy_1` and `get_strategy_2` methods from the `locations_extractor` object (an instance of the `LocationExtractor` class) to extract additional information based on the strategies.
11. If either "strategy_1" or "strategy_2" is not empty, add the current row to the output dataset using the `add_to_csv` function.
12. Save the output dataset to the specified "save_path" CSV file.

### Strategy 1:

In `strategy_1`, the algorithm aims to identify the counties and states associated with each US city mentioned in the articles. It follows the following steps:

1. Initialize empty dictionaries `result_county` and `result_state`.
2. Split the "counties" and "states" fields in the row into lists.
3. Iterate over each city in the "US_cities" list.
4. For each city, assign an empty string to the corresponding key in `result_state` and `result_county`.
5. Process the counties list:
   - If a county has multiple values (separated by commas), check if any of the values match a city in the "US_cities" list.
   - If a match is found, assign the matching county to the corresponding city in `result_county`.
   - If a county has a single value, assign that value directly to the corresponding city in `result_county`.
6. Process the states list in a similar manner as the counties list, assigning values to the corresponding city in `result_state`.
7. For each city in the "US_cities" list:
   - If both `result_county[location]` and `result_state[location]` have non-empty values:
     - Append the city, county, and state information to the "strategy_1" field in the row.
   - If only `result_county[location]` has a non-empty value:
     - Try to retrieve the state information for the county using the `df_simplemaps` dataset.
     - If successful, append the city, county, and state information to the "strategy_1" field in the row.
   - If only `result_state[location]` has a non-empty value:
     - Try to retrieve the county information for the state using the `df_simplemaps` dataset.
     - If successful, append the city, county, and state information to the "strategy_1" field in the row.

### Strategy 2:
In `strategy_2`, the algorithm further refines the county and state information for each US city mentioned in the articles. It follows the following steps:

1. Initialize empty dictionaries `result_county` and `result_state`.
2. Split the "counties" and "states" fields in the row into lists.
3. Iterate over each location in the "US_cities" list.
4. For each location, assign an empty list to the corresponding key in `result_county` and `result_state`.
5. Process the counties list:
   - Split the county information into a query and a list of counties.
   - Assign the list of counties to the corresponding query in `result_county`.
6. Process the states list in a similar manner as the counties list, assigning values to the corresponding query in `result_state`.
7. For each location in the "US_cities" list:
   - If both `result_county[location]` and `result_state[location]` have more than one county or state respectively:
     - Query the `df_simplemaps` dataset to retrieve county information for the location.
     - Sort the retrieved data based on population.
     - If the population of the first county is greater than 50,000:
       - Append the city, county name, and state name information to the "strategy_2" field in the row.
   - If `result_county[location]` has only one county and `result_state[location]` has more than one state:
     - Query the `df_simplemaps` dataset to retrieve the county information for the location and the specified county.
     - If successful, append the city, county, and state information to the "strategy_2" field in the row.
   - If `result_state[location]` has only one state and `result_county[location]` has more than one county:
     - Query the `df_simplemaps` dataset to retrieve the county information for the location and the specified state.
     - If successful, append the city, county, and state information to the "strategy_2" field in the row.
   - If both `result_state[location]` and `result_county[location]` have only one value:
     - Append the city, county, and state information to the "strategy_2" field in the row.

By applying these strategies, the algorithm attempts to extract more accurate and specific county and state information associated with each US city mentioned in the articles. The results are stored in the "strategy_1" and "strategy_2" fields of the dataset, allowing users to analyze and filter articles based on this refined geographical information.

In summary, this code reads a filtered dataset ("Filter 3") and applies strategies to extract additional information about counties and states associated with US cities. It filters out articles that do not have sufficient results in either "strategy_1" or "strategy_2" and saves the filtered data as a new dataset ("Filter 4") while preserving the "ID" column.

## Filter 5

The `get_gather_filter` function performs the following steps:

1. Reads the data from the "Filter 4" CSV file into a DataFrame, assuming the columns are labeled as: "ID", "URL", "TEXT", "TITLE", "DATE", "spacy_LOCATIONS", "flair_LOCATIONS", "DATES", "NUMBERS_CONTEXT", "NUMBERS", "GEOTEXT", "GEOTEXT_UNQ", "COUNTIES", "STATES", "STRATEGY 1", "STRATEGY 2".
2. Renames the columns of the DataFrame to ensure consistency.
3. Applies the `get_gathering_amount` method from the `NLPExtractor` class to the "NUMBERS_CONTEXT" column, which extracts the gathering amounts from the text.
     - The input text is converted to a list of lists using `ast.literal_eval` to parse the text as a nested list.
     - For each chunk (sublist) in the list, the function iterates over the words.
     - Each word is cleaned by removing any non-alphanumeric characters.
     - If the word's length is greater than 5, it is processed further.
     - The word is passed through the `nlp` (spaCy) pipeline for part-of-speech tagging and dependency parsing.
     - If the word ends with 's', the 's' is removed to handle plural forms.
     - The word is checked for certain conditions to determine if it represents a gathering amount.
     - The word is compared with the word "people" and other related terms using relative cosine similarity scores from the pre-trained GloVe model (`model_glove`).
     - If the conditions for a gathering amount are met, the word or words in the chunk are added to the `gathering_numbers` string.
     - The `gathering_numbers` string, containing the extracted gathering amounts, is returned.
5. Applies the `get_number` method from the `NLPExtractor` class to the "GATHERING_AMOUNT" column, which retrieves the numerical values from the extracted gathering amounts.
     - The input string is split into individual elements (gathering amounts) using a comma as the separator.
     - Each element is processed to extract the numerical values.
     - The function removes any non-alphanumeric characters from the element.
     - The resulting element is added to the `numbers` list if it represents a numerical value.
     - The `numbers` list, containing the extracted numerical values, is returned as a string, with the values separated by commas.
7. Applies the `identify_futuristic` method from the `NLPExtractor` class to each row in the DataFrame. This method identifies the tense (present or future) of the sentences containing the gathering amounts and adds a "TENSE/PT/FT" column to the DataFrame to indicate the tense and count of present and future tenses.
     - The sentences in the "TEXT" column are split into individual sentences using the `split_into_sentences` method from the `TextProcessing` class.
     - The sentences are filtered to include only those that contain any of the gathering amounts extracted in previous steps.
     - The filtered sentences are joined into a single string and assigned to the "GATHER_SENTS" column in the DataFrame.
     - The function initializes counters for present tense and future tense.
     - For each sentence in the filtered sentences:
       - The sentence is processed using the `nlp_sent` (spaCy) pipeline.
       - The function checks if the sentence contains tokens with specific morphological properties indicative of present or future tense.
       - Based on the presence of such tokens, the counters
9. Drops rows where the "GATHERING_NUMBER" column is empty (NaN) since they don't have valid gathering numbers.
10. Filters out rows where the "TENSE/PT/FT" column contains the word "future", indicating future tense gatherings.
11. Saves the filtered DataFrame as "Filter 5" in both CSV and Excel formats, preserving the "ID" column.

The purpose of this function is to perform filtering and preprocessing on the data, removing articles without gathering amounts and those with future tense gatherings, and storing the filtered data in "Filter 5" for further analysis or processing.
