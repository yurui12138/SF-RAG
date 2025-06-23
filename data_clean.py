import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import random
import logging
from config import Config
import requests
import logging
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class ImageTitleProcessor:
    @staticmethod
    def find_image_titles(lines, found_line_index):
        def check_title_and_images(direction):
            titles = []
            stop_flag = False
            for offset in range(1, 6):
                idx = found_line_index + direction * offset
                if not (0 <= idx < len(lines)):
                    break
                line = lines[idx].strip()
                if line.startswith("![](images/"):
                    stop_flag = True
                    break
                if line.lower().startswith(("table", "fig")):
                    titles.append((line, idx))
                    for inner_offset in range(1, 4):
                        new_idx = idx + direction * inner_offset
                        if not (0 <= new_idx < len(lines)) or lines[new_idx].strip().startswith("![](images/"):
                            stop_flag = True
                            break
                    break
            return titles, stop_flag

        titles_up, stop_up = check_title_and_images(-1)
        titles_down, stop_down = check_title_and_images(1)

        if not stop_up and titles_up:
            selected_title = titles_up[0]
        elif not stop_down and titles_down:
            selected_title = titles_down[0]
        elif titles_up and titles_down:
            selected_title = random.choice([titles_up[0], titles_down[0]])
        else:
            selected_title = titles_up[0] if titles_up else titles_down[0] if titles_down else None

        return [selected_title] if selected_title else []

    def update_img_titles(self, data, md_lines):
        for img in data:
            if isinstance(img["img_title"], list):
                img["img_title"] = [title.strip() for title in img["img_title"]
                                    if isinstance(title, str) and title.lower().startswith(('tab', 'fig'))]

        for idx, img in enumerate(data):
            if isinstance(img["img_title"], list) and len(img["img_title"]) > 1:
                prev_img = data[idx - 1] if idx > 0 else None
                next_img = data[idx + 1] if idx < len(data) - 1 else None
                if prev_img and not prev_img["img_title"] and prev_img["page_idx"] == img["page_idx"]:
                    prev_img["img_title"] = [img["img_title"][0].strip()]
                    del img["img_title"][0]
                if next_img and not next_img["img_title"] and next_img["page_idx"] == img["page_idx"]:
                    next_img["img_title"] = [img["img_title"][1].strip()]
                    del img["img_title"][1]

        for img in data:
            if not img["img_title"]:
                img_path = img["img_path"]
                found_line_index = next((i for i, line in enumerate(md_lines) if img_path in line), None)
                if found_line_index is not None:
                    titles = self.find_image_titles(md_lines, found_line_index)
                    if titles:
                        selected_title = min(titles, key=lambda x: abs(x[1] - found_line_index))
                        if selected_title and (selected_title[0].lower().startswith(('tab', 'fig'))):
                            img["img_title"] = [selected_title[0].strip()]

    def deduplicate_img_titles(self, data, md_lines):
        title_occurrences = {}
        for img in data:
            titles = img.get("img_title", [])
            if isinstance(titles, list) and titles:
                for title in titles:
                    stripped_title = title.strip()
                    title_occurrences.setdefault(stripped_title, []).append(img["img_path"])

        duplicates = {title: paths for title, paths in title_occurrences.items() if len(paths) > 1}

        if duplicates:
            for dup_title, img_paths in duplicates.items():
                for img_path in img_paths:
                    found_line_index = next((i for i, line in enumerate(md_lines) if img_path in line), None)
                    if found_line_index is not None:
                        titles = self.find_image_titles(md_lines, found_line_index)
                        if titles:
                            selected_title = min(titles, key=lambda x: abs(x[1] - found_line_index))
                            if selected_title and (selected_title[0].lower().startswith(('tab', 'fig'))):
                                for img in data:
                                    if img["img_path"] == img_path and dup_title in img["img_title"]:
                                        img["img_title"].remove(dup_title)
                                        img["img_title"].append(selected_title[0].strip())
                                        break

class ImageUploader:

    def __init__(self, picture_bed_token, picture_bed_url, max_retries=3, retry_delay=2):
        self.token = picture_bed_token
        self.upload_url = picture_bed_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def upload_image(self, image_path):
        attempt = 0
        while attempt < self.max_retries:
            try:
                with open(image_path, 'rb') as img_file:
                    files = {'file': img_file}
                    data = {'token': self.token}
                    response = requests.post(self.upload_url, data=data, files=files)
                    json_response = response.json()

                    if response.status_code == 200 and json_response.get('err') == 0 and 'url' in json_response:
                        return json_response['url']

                    logging.error(
                        f"Upload failed {image_path} ( {attempt + 1}/{self.max_retries}): {json_response}")

            except FileNotFoundError:
                logging.error(f"File not found：{image_path}")
                return None
            except requests.RequestException as e:
                logging.error(f"Upload failed ( {attempt + 1}/{self.max_retries})：{str(e)}")
            except Exception as e:
                logging.error(f"Upload failed ( {attempt + 1}/{self.max_retries})：{str(e)}")

            attempt += 1
            if attempt < self.max_retries:
                sleep(self.retry_delay)

        logging.error(f"Upload failed {image_path} ")
        return None


class FileProcessor:
    @staticmethod
    def extract_image_entries(data, results=None):
        if results is None:
            results = []

        def _extract(d):
            if isinstance(d, dict) and 'img_path' in d:
                entry = {
                    'img_path': d.get('img_path'),
                    'page_idx': d.get('page_idx', ''),
                    'img_title': d.get('img_title', [])
                }
                for key in ['table_caption', 'table_footnote', 'img_caption', 'img_footnote']:
                    if key in d and isinstance(d[key], list):
                        entry['img_title'].extend(d[key])
                results.append(entry)
            elif isinstance(d, (dict, list)):
                for value in d.values() if isinstance(d, dict) else d:
                    _extract(value)

        _extract(data)
        return results

    @staticmethod
    def get_unique_file(folder, extension, error_msg):
        files = [f for f in os.listdir(folder) if f.endswith(extension) and not f.startswith('Summarize')]
        if len(files) != 1:
            logging.error(error_msg)
            return None
        return os.path.join(folder, files[0])

    @staticmethod
    def shorten_filename(filename, length=10):
        name, ext = os.path.splitext(filename)
        return f"{name[:length]}{ext}"


class ImageDataProcessor:

    def __init__(self, config: Config):
        self.config = config
        self.title_processor = ImageTitleProcessor()
        self.uploader = ImageUploader(picture_bed_token=self.config.picture_bed_token, picture_bed_url=self.config.picture_bed_url)
        self.file_processor = FileProcessor()

    def process_images(self, root_folder):
        images_folder = next(
            (os.path.join(dirpath, 'images') for dirpath, dirs, _ in os.walk(root_folder) if 'images' in dirs), None)

        if not images_folder:
            logging.error("No images folder")
            return None, None, None

        parent_dir = os.path.dirname(images_folder)
        md_file = self.file_processor.get_unique_file(parent_dir, '.md', "No md file")
        json_file = self.file_processor.get_unique_file(parent_dir, '_content_list.json', "No _content_list.json file")

        if not md_file or not json_file:
            return None, None, None

        for filename in os.listdir(images_folder):
            if filename.lower().endswith('.jpg'):
                old_filepath = os.path.join(images_folder, filename)
                new_filename = self.file_processor.shorten_filename(filename)
                new_filepath = os.path.join(images_folder, new_filename)
                if new_filename != filename:
                    os.rename(old_filepath, new_filepath)
                    self._update_file_references(md_file, filename, new_filename)
                    self._update_file_references(json_file, filename, new_filename)

        return json_file, md_file, parent_dir

    def _update_file_references(self, file_path, old_name, new_name):
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            updated_content = content.replace(old_name, new_name)
            file.seek(0)
            file.write(updated_content)
            file.truncate()

    def process_data(self, input_file, parent_dir, output_file, md_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            extracted_data = self.file_processor.extract_image_entries(data)
            with open(md_file, 'r', encoding='utf-8') as md_f:
                md_lines = md_f.readlines()

            self.title_processor.update_img_titles(extracted_data, md_lines)
            self.title_processor.deduplicate_img_titles(extracted_data, md_lines)

            def process_item(item):
                if not item["img_title"]:
                    # return item
                    item["img_title"] = '该图片没有标题'
                absolute_path = os.path.abspath(os.path.join(parent_dir, item["img_path"]))
                url = self.uploader.upload_image(absolute_path)
                if url:
                    item["img_path"] = url
                return item

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_item, data) for data in extracted_data]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Task execution exception: {str(e)}")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=4)

            # logging.info(f"Data successfully written to {output_file}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

class DataClean:
    def __init__(self, doc_path: str, json_file_path: str):
        self.doc_path = doc_path
        self.json_file_path = json_file_path
        self.docs_text = self._load_document()
        self.abstract_info = self._find_abstract()

    def _read_file(self, file_path: str) -> list[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()

    def _remove_special_lines_and_add_spaces(self, lines: list[str]) -> list[str]:
        lines = [line for line in lines if line.strip() != ""]
        new_lines = []
        for i, line in enumerate(lines):
            if line.startswith('#') or line.startswith('![]'):
                new_lines.append("\n")
            new_lines.append(line)
        return new_lines

    def _remove_special_lines(self, lines: list[str]) -> list[str]:
        return [line for line in lines if not (line.startswith('![]') or line.startswith('<html><body><table>'))]

    def _process_md_with_json(self, md_lines: list[str], json_data: list) -> list[str]:
        for item in json_data:
            img_title = item.get('img_title')[0]
            for i, line in enumerate(md_lines):
                if img_title in line:
                    del md_lines[i]
                    break
        return md_lines

    def _load_document(self) -> str:
        md_lines = self._read_file(self.doc_path)
        md_lines = self._remove_special_lines(md_lines)
        md_lines = self._remove_special_lines_and_add_spaces(md_lines)

        with open(self.json_file_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        md_lines = self._process_md_with_json(md_lines, json_data)
        processed_text = ''.join(md_lines)
        return processed_text.replace('{', '{{').replace('}', '}}')

    def _find_abstract(self) -> str:
        lines = [line.strip() for line in self.docs_text.splitlines()]

        if abstract := self._find_by_header(lines, '## abstract', return_next_line=True):
            return abstract
        if abstract := self._find_by_header(lines, 'abstract', return_next_line=False):
            return abstract
        if abstract := self._find_by_introduction(lines):
            return abstract
        return self.docs_text[:5000]

    @staticmethod
    def _find_by_header(lines: list[str], header: str, return_next_line: bool) -> str:
        header = header.lower()
        for i, line in enumerate(lines):
            if line.lower().startswith(header):
                if not return_next_line:
                    return line
                for candidate in lines[i + 1:]:
                    if candidate:
                        return candidate
        return ''

    @staticmethod
    def _find_by_introduction(lines: list[str]) -> str:
        for i, line in enumerate(lines):
            if line.lower().endswith('introduction') or line.lower().startswith('## introduction'):
                content = []
                count = 0
                for candidate in lines[i + 1:]:
                    if candidate:
                        content.append(candidate)
                        count += 1
                        if count >= 5:
                            return ' '.join(content)
        return ''

def data_clean(root_directory, config):
    processor = ImageDataProcessor(config)
    json_file, md_file, parent_dir = processor.process_images(root_directory)
    if all([json_file, md_file, parent_dir]):
        processor.process_data(json_file, parent_dir, os.path.join(parent_dir, 'images.json'), md_file)

    file_processor = DataClean(doc_path=md_file, json_file_path=os.path.join(parent_dir, 'images.json'))
    return file_processor
