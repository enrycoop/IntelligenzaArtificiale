// IntelligenzaArtificiale.cpp: definisce il punto di ingresso dell'applicazione.
//

/** Relevant Information:
   --- This is perhaps the best known database to be found in the pattern
       recognition literature.  Fisher's paper is a classic in the field
       and is referenced frequently to this day.  (See Duda & Hart, for
       example.)  The data set contains 3 classes of 50 instances each,
       where each class refers to a type of iris plant.  One class is
       linearly separable from the other 2; the latter are NOT linearly
       separable from each other.
   --- Predicted attribute: class of iris plant.
   --- This is an exceedingly simple domain.
   --- This data differs from the data presented in Fishers article
	(identified by Steve Chadwick,  spchadwick@espeedaz.net )
	The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
	where the error is in the fourth feature.
	The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
	where the errors are in the second and third features. 
Number of Instances: 150 (50 in each of three classes)

Number of Attributes: 4 numeric, predictive attributes and the class

Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class:
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

Missing Attribute Values: None

Summary Statistics:
           Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

Class Distribution: 33.3% for each of 3 classes.


NOTE: for the experiments Iris virginica was removed
*/

#include "IntelligenzaArtificiale.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

double categoryTodouble(std::string s)
{
  if ("Iris-setosa" == s)
    return 1.0;
  else
    return -1.0;
}

std::string doubleToCategory(double f)
{
  if (1.0 == f)
    return "Iris-setosa";
  else
    return "Iris-versicolor";
}

template<typename T>
void stampa(vector<T> X)
{
  cout << "[";
  for (auto& x : X) cout << x << " ";
  cout << "]\n";
}

void stampa_campioni(std::vector<std::vector<double>> X)
{
  for (auto& x : X)
    stampa(x);
}



int main()
{
  const std::string FOLDER_PATH{ "../../../data/iris.data" };

  std::ifstream file(FOLDER_PATH);

  if (!file.is_open()) {
    std::cerr << "Errore nell'aprire il file" << std::endl;
    exit(-1);
  }
  std::vector<std::vector<double>> X{};
  std::vector<double> y{};

  std::vector<std::vector<double>> X_test{};
  std::vector<double> y_test{};

  std::string linea;

  int i = 0;
  while (std::getline(file, linea)) {
    std::vector<double> xi{};
    std::stringstream sstream(linea);
    std::string campo;
    while (std::getline(sstream, campo, ','))
    {
      try {
        xi.push_back(std::stof(campo));
      }
      catch (const std::invalid_argument) {
        if (i > 40 && i < 60)
          y_test.push_back(categoryTodouble(campo));
        else
          y.push_back(categoryTodouble(campo));
      }
    }
    if (i > 40 && i < 60)
      X_test.push_back(xi);
    else
      X.push_back(xi);
    ++i;
  }

  //stampa_campioni(X);
  //stampa(y);

  unique_ptr<Model> model = make_unique<Perceptron>(0.1, 10);

  model->fit(X, y, true);

  for (int j=0; j < X_test.size(); ++j)
  {
    //std::string pred = doubleToCategory(model.predict(X_test.at(j)));
    //std::string cat = doubleToCategory(y.at(j));

    double pred = model->predict(X_test.at(j));
    double cat = y_test.at(j);
    cout << "prediction: " << pred << " True class: " << 
      cat << (pred==cat ? " OK" : " WRONG") << endl;
  }
    
	//cout << "prediction: " << model.predict(iris) << endl;
	return 0;
}
