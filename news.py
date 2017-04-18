def write_result(size):
	text_scanner = open("newsCorpora_full.txt", "r")
	csv_scanner = open("newsCorpora.csv", "r")
	invalid_indices = [6, 82, 84, 86, 94]
	b = []
	t = []
	e = []
	m = []

	# the id of the current news
	news_id = -1

	lines = text_scanner.readlines()
	info_lines = csv_scanner.readlines()
	current_news = ""
	for line in lines:
		# beginning of a news
		if ">>>>" in line:
			# remove leading and trailing whitespace
			stripped = line.strip()
			news_id = int(line[4:])
			current_news = ""
		# end of a news
		elif "<<<<" in line:
			if len(current_news) == 0:
				continue
			# extract information from csv file based on the news_id
			# the type is on the 5th column
			news_type = info_lines[news_id].split("\t")[4]
			# update the ids
			news_id = -1
			# add the news to the appropriate list
			add_news(current_news, news_type, size, b, t, e, m)
		elif news_id not in invalid_indices:
			current_news = current_news + line

		if len(b) >= size and len(t) >= size and len(e) >= size and len(m) >= size:
			break

	file_writer = open("sl" + str(size * 4) + ".txt", "w+")
	print_list(file_writer, b, "b")
	print_list(file_writer, t, "t")
	print_list(file_writer, e, "e")
	print_list(file_writer, m, "m")

	text_scanner.close()
	csv_scanner.close()
	file_writer.close()


def add_news(news, news_type, size, b, t, e, m):
	if news_type == "b":
		if len(b) < size:
			b.append(news)
	elif news_type == "t":
		if len(t) < size:
			t.append(news)
	elif news_type == "e":
		if len(e) < size:
			e.append(news)
	elif news_type == "m":
		if len(m) < size:
			m.append(news)
	else:
		print("unrecognized type " + news_type)


def print_list(writer, list, title):
	writer.write("***** " + title + " *****\n")
	for i in range(len(list)):
		writer.write(">>>>" + str(i+1) + "\n" + list[i] + "<<<<" + str(i+1) + "\n");
	writer.write("*****\n");


if __name__ == '__main__':
    write_result(180)

