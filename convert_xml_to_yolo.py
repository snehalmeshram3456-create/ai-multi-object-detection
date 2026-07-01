import os
import xml.etree.ElementTree as ET

annotations_dir = "dataset/annotations"
labels_dir = "dataset/valid/labels"

os.makedirs(labels_dir, exist_ok=True)

for xml_file in os.listdir(annotations_dir):

    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotations_dir, xml_file))
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    output_file = os.path.join(
        labels_dir,
        xml_file.replace(".xml", ".txt")
    )

    with open(output_file, "w") as f:

        for obj in root.findall("object"):

            class_id = 0

            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            f.write(
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{box_width} "
                f"{box_height}\n"
            )

print("Conversion Complete")