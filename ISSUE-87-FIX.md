## What the fix does:

* **_on_download**  - creates an asynchrone task that uses the function _save_download and appends the asynchrone event to a set in self._download_tasks

---

* **_save_download**  - handels the download name so there is no path traversal and changes inapropriate characters to _

* **_save_download**  - function saves files to paths.files_dir() / "downloads", so the apps config folder is used, where at the end an relative path is returned for the saved file

* **_save_download**  - Prevents overwriting files with same name by checking the file names and trying to append a number to the file name until file name is unique

* **_save_download**  - Uses a lock (self._download_lock) for saving files, so there will be only one access to the downloads directory at a time, preventing corrupted files or direcotry

---

* **_include_downloads**  - is waiting for the downloads to finish. It is called before returning results from functions - open, click, type, select, press, back

* **_include_downloads**  - Because the browser activity is asynchronous, this function gets previous not finished download list and substracts the previouse/current asynchrone tasks to see, if new downloads appeared within the time, that is set in DOWNLOAD_APPEAR_GRACE_S

* **_include_downloads**  - If a new download appeared, it waits until it finishes with **await asyncio.gather(*tasks)** and then removes the tasks from the set

---

* **test_download_is_saved_and_reported** - test uses offline site and lets the browser download data through finding link on the attribute on the site. 
1. assert - It expects from the browser one download at the expected path
2. assert - checks the saved content is equal to the content on the site
