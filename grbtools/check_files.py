import pandas as pd

df1 = pd.read_excel("../data_sources/batse_3July2023.xls")  # Read the first Excel file
df2 = pd.read_excel("../data_sources/batse_catalog.xlsx.xls")  # Read the second Excel file

# drop irrelevant columns from df2
df2.drop

# get triggers from both files
triggers1 = set(df1['trigger_num'])
triggers2 = set(df2['trigger_num'])
# combine the two sets
triggers = triggers1.union(triggers2)

# for each trigger, compare the two files
for trigger in triggers:
    # get the rows for the current trigger from both files
    rows1 = df1[df1['trigger_num'] == trigger]
    rows2 = df2[df2['trigger_num'] == trigger]
    
    # make sure that rows contains only 1 row for each file
    assert len(rows1) == 1
    assert len(rows2) == 1





def compare_excel_files(file1, file2):
    

    # Get the unique trigger_nums from both files
    trigger_nums1 = set(df1['trigger_num'])
    trigger_nums2 = set(df2['trigger_num'])

    # Find the trigger_nums that do not match
    unmatched_trigger_nums = trigger_nums1.symmetric_difference(trigger_nums2)

    return unmatched_trigger_nums

# Provide the paths to the two Excel files
file1_path = 'path_to_file1.xlsx'
file2_path = 'path_to_file2.xlsx'

unmatched_trigger_nums = compare_excel_files(file1_path, file2_path)

print("Trigger_nums that do not match:")
for trigger_num in unmatched_trigger_nums:
    print(trigger_num)




import pandas as pd

# Read the first Excel file
file1 = pd.read_excel('file1.xlsx')
# Read the second Excel file
file2 = pd.read_excel('file2.xlsx')

# Compare the two files
mismatched_trigger_nums = []
for index, row in file1.iterrows():
    trigger_num = row['trigger_num']
    matching_row = file2[file2['trigger_num'] == trigger_num]
    if len(matching_row) == 0 or not row.equals(matching_row.iloc[0]):
        mismatched_trigger_nums.append(trigger_num)

# Print the mismatched trigger_nums
print("Mismatched trigger_nums:")
for trigger_num in mismatched_trigger_nums:
    print(trigger_num)