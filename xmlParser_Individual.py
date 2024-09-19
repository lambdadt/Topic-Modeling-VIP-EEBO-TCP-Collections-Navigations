import xml.etree.ElementTree as ET
import os

# Define file paths
input_file = '/Users/parag/Desktop/Topic-Modeling-VIP-EEBO-TCP-Collections-Navigations/Navigations_headed_xml/A0-A5/A00007.headed.xml'
output_text_file = '/Users/parag/Desktop/Topic-Modeling-VIP-EEBO-TCP-Collections-Navigations/Navigations_headed_xml/Parsed_texts/A00007_parsed_text.txt'
output_footnotes_file = '/Users/parag/Desktop/Topic-Modeling-VIP-EEBO-TCP-Collections-Navigations/Navigations_headed_xml/Parsed_texts/A00007_footnotes.txt'

def parse_xml(input_file):
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Initialize variables to store text and footnotes
    text_content = []
    footnotes = []

    # Recursive function to extract text and footnotes
    def extract_content(element):
        for child in element:
            if child.tag.lower() in ['note', 'footnote', 'ref', 'fn']:  # Assuming footnotes are in these tags
                footnotes.append(child.text.strip() if child.text else '')
                extract_content(child)
            else:
                if child.text:
                    text_content.append(child.text)
                extract_content(child)  # Recurse into child elements
            if child.tail:
                text_content.append(child.tail)

    # Start extraction
    extract_content(root)

    return text_content, footnotes

def save_to_file(filename, content):
    # Create the file if it doesn't exist and write content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))

# Ensure the output files are created if they don't exist
if not os.path.exists(output_text_file):
    open(output_text_file, 'w').close()

if not os.path.exists(output_footnotes_file):
    open(output_footnotes_file, 'w').close()

# Parse the XML file
text_content, footnotes = parse_xml(input_file)

# Save the parsed text and footnotes to separate files
save_to_file(output_text_file, text_content)
save_to_file(output_footnotes_file, footnotes)

print(f'Text content saved to: {output_text_file}')
print(f'Footnotes saved to: {output_footnotes_file}')
