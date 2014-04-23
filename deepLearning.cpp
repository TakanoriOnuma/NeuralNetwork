#include <iostream>
#include <vector>

using namespace std;

class Neuron {
    double u;
    double x;
    vector<double> w;
    double bias;

public:
    Neuron(const vector<double>& w, double bias)
        : w(w), bias(bias), u(0.0), x(0.0)
    {
    }

    // --- getter --- //
    double getU() const { return u; }
    double getX() const { return x; }
    vector<double>& getW() { return w; }
    double getBias() const { return bias; }
};

double scalarProduct(const vector<double>& a, const vector<double>& b)
{
    double sum = 0.0;
    if(a.size() == b.size()) {
        for(int i = 0; i < a.size(); i++) {
            sum += a[i] * b[i];
        }
    }
    return sum;
}

int main()
{
    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    neurons.resize(5);
    neurons[0].resize(1);
    neurons[1].resize(5);
    neurons[2].resize(3);
    neurons[3].resize(5);
    neurons[4].resize(1);

    vector<double> w(1);
    w[0] = 1.0;
    neurons[0][0] = new Neuron(w, 0.0);

    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i - 1].size(); j++) {
            w[j] = 1.0;
        }
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j] = new Neuron(w, 0.0);
        }
    }

    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:" << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << endl;
        }
    }


    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}