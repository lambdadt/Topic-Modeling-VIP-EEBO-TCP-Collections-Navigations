import xml.etree.ElementTree as ET
import os
import pandas as pd

# Define the directory containing the XML files
xml_directory = '/Users/parag/Desktop/Topic-Modeling-VIP-EEBO-TCP-Collections-Navigations/Navigations_headed_xml/A0-A5/'
output_directory = '/Users/parag/Desktop/Topic-Modeling-VIP-EEBO-TCP-Collections-Navigations/Navigations_headed_xml/Parsed_texts/'
file_prefix = 'A'
file_suffix = '.headed.xml'
spreadsheet_output_directory = '/Users/parag/Desktop/Topic-Modeling-VIP-EEBO-TCP-Collections-Navigations/Navigations_headed_xml/'
spreadsheet_file = os.path.join(spreadsheet_output_directory, 'eebo_data.xlsx')

def extract_metadata(xml_filepath):
    """Extract EEBO ID, ESTC number, Title, and Author from the XML file."""
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        eebo_id = None
        estc_number = None
        title = None
        author = None

        # Search for EEBO ID, ESTC number, Title, and Author
        header = root.find('.//HEADER')
        if header is not None:
            # Extract EEBO ID
            eebo_id_tag = header.find('.//IDNO[@TYPE="DLPS"]')
            if eebo_id_tag is not None:
                eebo_id = eebo_id_tag.text.strip()


            # Extract the second STC number which is ESTC (STC tag is present twice)
            stc_tags = header.findall('.//IDNO[@TYPE="stc"]')
            if len(stc_tags) > 1:
                full_estc_text = stc_tags[1].text.strip()  # Take the second occurrence
                # Extract only the part after "ESTC "
                estc_number = full_estc_text.split('ESTC ')[-1]

            # Extract Title
            title_tag = header.find('.//TITLESTMT/TITLE[@TYPE="245"]')
            if title_tag is not None:
                title = title_tag.text.strip()

            # Extract Author
            author_tag = header.find('.//AUTHOR')
            if author_tag is not None:
                author = author_tag.text.strip()

        return eebo_id, estc_number, title, author

    except ET.ParseError as e:
        print(f"Failed to parse {xml_filepath}: {e}")
        return None, None, None, None

def process_files(start_num, end_num):
    """Process files from start_num to end_num and create a spreadsheet with metadata."""
    data = []
    num = start_num

    while num <= end_num:
        # Construct the file name
        num_str = str(num).zfill(5)
        xml_filename = f"{file_prefix}{num_str}{file_suffix}"
        xml_filepath = os.path.join(xml_directory, xml_filename)

        # Check if the file exists
        if os.path.exists(xml_filepath):
            print(f"Processing file: {xml_filename}")
            # Extract metadata from the XML file
            eebo_id, estc_number, title, author = extract_metadata(xml_filepath)
            if eebo_id:
                data.append({
                    'EEBO ID': eebo_id,
                    'ESTC Number': estc_number,
                    'Title': title,
                    'Author': author
                })
        else:
            print(f"File not found: {xml_filename}")

        num += 1

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    os.makedirs(output_directory, exist_ok=True)  # Ensuring the output directory exists

    # Save the DataFrame to an Excel file
    df.to_excel(spreadsheet_file, index=False)

    print(f"Spreadsheet saved to: {spreadsheet_file}")

# Start processing from A00005 to A60000
process_files(5, 60000)