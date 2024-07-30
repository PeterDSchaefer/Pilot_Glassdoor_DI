########################################################################################################################
### READ JSON FILES WITH GLASSDOOR REVIEWS
########################################################################################################################

# This code reads all JSON files with Glassdoor reviews in one folder. It collects all review texts (pros, cons, recommendations)
# of one company in one text file. Then, it reads the ratings themselves and collects them in a csv file.

import os
import json
from collections import defaultdict
import pandas as pd

# Define the path to the folder containing the JSON files
base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, '03_scraped_gd_reviews',)
output_folder_path = os.path.join(base_dir, '04_processed_gd_reviews', '01_txt')
dta_folder_path = os.path.join(base_dir, '04_processed_gd_reviews', '02_dta')

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter the list to include only JSON files
json_files = [file for file in file_list if file.endswith('.json')]

# Iterate through each JSON file in the list
for json_file in json_files:
    # Construct the full path to the JSON file
    file_path = os.path.join(folder_path, json_file)

    # Read the JSON file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)

            # Dictionary to store reviews by year
            reviews_by_year = defaultdict(list)
            all_reviews = []

            # Check if 'reviews' is in the JSON data
            if 'reviews' in data:
                # Iterate through each review in the reviews list
                for review in data['reviews']:
                    # Extract the year from reviewDateTime
                    review_date_time = review.get('reviewDateTime', '')
                    if review_date_time:
                        year = review_date_time.split('-')[0]

                        # Extract the desired fields if they exist
                        cons = review.get('cons', '')
                        advice = review.get('advice', '')
                        pros = review.get('pros', '')

                        # Formatted review text
                        review_text = f"Cons: {cons}\nAdvice: {advice}\nPros: {pros}\n"

                        # Append the review to the list for the corresponding year
                        reviews_by_year[year].append(review_text)

                        # Append the review to the all_reviews list
                        all_reviews.append(review_text)

            # Write reviews to separate text files for each year
            for year, reviews in reviews_by_year.items():
                # Create the output file name by replacing the .json extension with _<year>.txt
                output_file_path = os.path.join(output_folder_path, os.path.splitext(json_file)[0] + f'_{year}.txt')

                # Open the output text file in write mode
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    # Write all reviews for the current year to the file
                    output_file.write("\n".join(reviews))

            # Write all reviews to a single text file
            all_reviews_file_path = os.path.join(output_folder_path, os.path.splitext(json_file)[0] + '_all_reviews.txt')
            with open(all_reviews_file_path, 'w', encoding='utf-8') as all_reviews_file:
                all_reviews_file.write("\n".join(all_reviews))

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {json_file}: {e}")



####### Export individual variables #######

# List to store data for DataFrame
data = []

# Iterate through each JSON file in the list
for json_file in json_files:
    # Construct the full path to the JSON file
    file_path = os.path.join(folder_path, json_file)

    # Extract the company name from the file name (without the .json extension)
    company_name = os.path.splitext(json_file)[0]

    # Read the JSON file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data_json = json.load(file)

            # Check if 'reviews' is in the JSON data
            if 'reviews' in data_json:
                # Iterate through each review in the reviews list
                for review in data_json['reviews']:
                    # Extract the desired fields
                    ratingBusinessOutlook = review.get('ratingBusinessOutlook', None)
                    ratingCareerOpportunities = review.get('ratingCareerOpportunities', None)
                    ratingDiversityAndInclusion = review.get('ratingDiversityAndInclusion', None)
                    ratingOverall = review.get('ratingOverall', None)
                    lengthOfEmployment = review.get('lengthOfEmployment', None)
                    #jobTitle = review.get('jobTitle', None)
                    reviewDateTime = review.get('reviewDateTime', None)

                    # Append the extracted data to the list as a dictionary
                    data.append({
                        'company_name': company_name,
                        'ratingBusinessOutlook': ratingBusinessOutlook,
                        'ratingCareerOpportunities': ratingCareerOpportunities,
                        'ratingDiversityAndInclusion': ratingDiversityAndInclusion,
                        'ratingOverall': ratingOverall,
                        'lengthOfEmployment': lengthOfEmployment,
                        #'jobTitle': jobTitle,
                        'reviewDateTime': reviewDateTime
                    })

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {json_file}: {e}")

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Define the path for the output .dta file
output_dta_file_path = os.path.join(dta_folder_path, 'reviews_data.dta')

# Export the DataFrame to a Stata .dta file
df.to_stata(output_dta_file_path, write_index=False)

print(f"Data successfully exported to {output_dta_file_path}")