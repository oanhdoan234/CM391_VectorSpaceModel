import java.util.*;
import java.io.*;

public class News {

 public static final int DEFAULT_SIZE = 250;

 public static void main( String[] args ) {
  try {
   Scanner textSC = new Scanner(new File("newsCorpora_full.txt"));
   Scanner csvSC = new Scanner(new File("newsCorpora.csv"));
   int[] invalidIndices = {6, 82, 84, 86, 94};
   ArrayList<String> b = new ArrayList<String>();
   ArrayList<String> t = new ArrayList<String>();
   ArrayList<String> e = new ArrayList<String>();
   ArrayList<String> m = new ArrayList<String>();

   // extract news
   StringBuilder builder = new StringBuilder();
   int id = -1;
   int previousId = 0;
   while (textSC.hasNextLine()) {
    String line = textSC.nextLine();
    // beginning of a news
    if (line.indexOf(">>>>") >= 0) {
     builder = new StringBuilder();
     id = Character.getNumericValue(line.charAt(line.length()-1));
    }
    else if (line.indexOf("<<<<") >= 0) {
     String news = builder.toString();
     if (news.length() == 0) continue;
     String[] newsInfo = findInfo(csvSC, id, previousId);
     previousId = id;
     id = -1;
     if (newsInfo.length < 8) continue;
     String type = newsInfo[4];
     if ( type.equals("b") && b.size() < DEFAULT_SIZE ) {
      b.add(news);
     }
     else if ( type.equals("t") && t.size() < DEFAULT_SIZE ) {
      t.add(news);
     }
     else if ( type.equals("e") && e.size() < DEFAULT_SIZE ) {
      e.add(news);
     }
     else if ( type.equals("m") && m.size() < DEFAULT_SIZE ) {
      m.add(news);
     }
     else {
      System.err.println("unrecognized type: " + type);
     }
    }
    else if (!contains(invalidIndices, id)) {
     builder.append(line + "\n");
    }
    if (b.size() >= DEFAULT_SIZE && t.size() >= DEFAULT_SIZE && e.size() >= DEFAULT_SIZE && m.size() >= DEFAULT_SIZE)
     break;
   }
   FileWriter writer = new FileWriter(new File("shortlist.txt"));
   printArr(writer, b, "b");
   printArr(writer, t, "t");
   printArr(writer, e, "e");
   printArr(writer, m, "m");
   writer.flush();
   writer.close();
   textSC.close();
  } catch (Exception e) {
   e.printStackTrace();
  }
 }


 public static String[] findInfo(Scanner sc, int newsId, int previousId) {

  String line = "";
  for (int i = previousId ; i < newsId ; i++) {
   line = sc.nextLine();
   if ( !sc.hasNextLine() ) break;
  }
  return line.split("\t");
 }


 public static boolean contains( int[] arr, int x ) {
  for ( int y : arr ) {
   if ( y == x ) return true;
  }
  return false;
 }


 public static void printArr(FileWriter writer, ArrayList<String> arr, String title) {
  try {
   writer.write("***** " + title + " *****\n");
   for ( int i = 0 ; i < arr.size() ; i++ ) {
    writer.write(">>>>" + (i+1) + "\n" + arr.get(i) + "<<<<" + (i+1) + "\n");
   }
   writer.write("*****\n");
  } catch (Exception e) {
   e.printStackTrace();
  }
 }
}
