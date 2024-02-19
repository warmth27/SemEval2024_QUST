import json

def convert_to_desired_format(chatgpt_answers, entry_id):
    if chatgpt_answers:
        text = chatgpt_answers[0]
        label = 0
    else:
        # Handle the case when neither human_answers nor chatgpt_answers is present
        text = ""
        label = -1

    output = {
        "text": text,
        "label": label,  # Adjusting label to start from 1 and increment
        "model": "human" if label == 0 else "chatgpt",
        "id": entry_id  # You may replace this with an appropriate identifier
    }

    return output

def process_jsonl_file(input_file_path, output_file_path):
    formatted_data = []
    entry_id = 3933  # Starting entry_id from 1

    with open(input_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            chatgpt_answers = data.get("human_answers", [])

            output = convert_to_desired_format(chatgpt_answers, entry_id)
            formatted_data.append(output)
            entry_id += 1

    with open(output_file_path, 'w') as output_file:
        json.dump(formatted_data, output_file, indent=4)

# Example usage:
input_jsonl_file_path = "finance.jsonl"
output_jsonl_file_path = "dev.json"
process_jsonl_file(input_jsonl_file_path, output_jsonl_file_path)
# import json
#
# def process_single_entry(entry, entry_id):
#     entry["id"] = entry_id
#     return entry
#
# def process_jsonl_file(input_file_path, output_file_path):
#     formatted_data = []
#     entry_id = 0  # Starting entry_id from 0
#
#     with open(input_file_path, 'r') as file:
#         for line in file:
#             try:
#                 data = json.loads(line)
#                 formatted_entry = process_single_entry(data, entry_id)
#                 formatted_data.append(formatted_entry)
#                 entry_id += 1
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON: {e}")
#
#     with open(output_file_path, 'w') as output_file:
#         for entry in formatted_data:
#             json.dump(entry, output_file, indent=4)
#             output_file.write('\n')
#
# # Example usage:
# input_jsonl_file_path = "dev_hum.jsonl"
# output_jsonl_file_path = "dev.jsonl"
# process_jsonl_file(input_jsonl_file_path, output_jsonl_file_path)


