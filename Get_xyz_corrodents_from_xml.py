"""
This is a quick and dirty script that extracts xyz coordinates from a xml file.
It is currently set up for HCP_MMP atlas. Any files used can be found at:
https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL

"""

import xml.etree.ElementTree as ET
import numpy as np

def parse_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    coordinates = []

    # Iterate over 'person' elements
    data_element = root.find('data')
    if data_element is not None:
        for label in data_element.findall('label'):
            # Extract attributes
            x = label.get('x')
            y = label.get('y')
            z = label.get('z')
            coordinates.append((float(x), float(y), float(z)))
    return coordinates

if __name__ == "__main__":
    file_path = r'C:\Users\em17531\Desktop\HCP-Multi-Modal-Parcellation-1.0.xml'  # Change this as needed
    coordinates = parse_xml(file_path)
    del coordinates[0]
    xyz_loc_array = np.stack(coordinates)
