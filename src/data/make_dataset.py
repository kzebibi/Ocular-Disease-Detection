# import os
# import shutil


# def process_data(raw_data_path, processed_data_path):
#     # Create directories for processed data
#     train_dir = os.path.join(processed_data_path, "Train")
#     test_dir = os.path.join(processed_data_path, "Test")
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     # Process train data
#     source_train_dir = os.path.join(raw_data_path, "DATASET_101", "Train")
#     process_set(source_train_dir, train_dir)

#     # Process test data
#     source_test_dir = os.path.join(raw_data_path, "DATASET_101", "Test")
#     process_set(source_test_dir, test_dir)


# def process_set(source_dir, target_dir):
#     for label in os.listdir(source_dir):
#         label_dir = os.path.join(source_dir, label)
#         if os.path.isdir(label_dir):
#             target_label_dir = os.path.join(target_dir, label)
#             os.makedirs(target_label_dir, exist_ok=True)

#             for image_filename in os.listdir(label_dir):
#                 source_path = os.path.join(label_dir, image_filename)
#                 target_path = os.path.join(target_label_dir, image_filename)
#                 shutil.copy(source_path, target_path)


# def main():
#     raw_data_path = os.path.join("data", "raw")
#     processed_data_path = os.path.join("data", "processed")

#     process_data(raw_data_path, processed_data_path)
#     print("Data processing completed.")


# if __name__ == "__main__":
#     main()
