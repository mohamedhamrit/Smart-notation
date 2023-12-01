# Smart-notation
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_set>
// Function to calculate TF (Term Frequency)
std::unordered_map<std::string, int> calculateTF(const std::vector<std::string>& document) {
    std::unordered_map<std::string, int> termFrequency;

    for (const auto& term : document) {
        termFrequency[term]++;
    }

    return termFrequency;
}

// Function to calculate IDF (Inverse Document Frequency)
std::unordered_map<std::string, double> calculateIDF(const std::vector<std::vector<std::string>>& documents) {
    std::unordered_map<std::string, double> inverseDocumentFrequency;
    std::unordered_map<std::string, int> documentFrequency;

    for (const auto& document : documents) {
        std::unordered_set<std::string> uniqueTerms(document.begin(), document.end());

        for (const auto& term : uniqueTerms) {
            documentFrequency[term]++;
        }
    }

    for (const auto& pair : documentFrequency) {
        inverseDocumentFrequency[pair.first] = log(static_cast<double>(documents.size()) / pair.second);
    }

    return inverseDocumentFrequency;
}

// Function to calculate TF-IDF
std::vector<double> calculateTFIDF(const std::vector<std::string>& document, const std::unordered_map<std::string, double>& idf) {
    std::vector<double> tfidfVector;

    std::unordered_map<std::string, int> tf = calculateTF(document);

    for (const auto& pair : tf) {
        double tfidf = pair.second * idf.at(pair.first);
        tfidfVector.push_back(tfidf);
    }

    return tfidfVector;
}

// Function to calculate cosine similarity
double cosineSimilarity(const std::vector<double>& queryVector, const std::vector<double>& documentVector) {
    double dotProduct = 0.0;
    double queryVectorMagnitude = 0.0;
    double documentVectorMagnitude = 0.0;

    for (size_t i = 0; i < queryVector.size(); ++i) {
        dotProduct += queryVector[i] * documentVector[i];
        queryVectorMagnitude += queryVector[i] * queryVector[i];
        documentVectorMagnitude += documentVector[i] * documentVector[i];
    }

    queryVectorMagnitude = sqrt(queryVectorMagnitude);
    documentVectorMagnitude = sqrt(documentVectorMagnitude);

    if (queryVectorMagnitude == 0 || documentVectorMagnitude == 0) {
        return 0.0;
    }

    return dotProduct / (queryVectorMagnitude * documentVectorMagnitude);
}

int main() {
    std::vector<std::vector<std::string>> documents = {
        {"apple", "orange", "banana", "kiwi"},
        {"orange", "pear", "banana", "grape"},
        {"apple", "banana", "orange"}
        // Add more documents as needed
    };

    std::vector<std::string> query = {"apple", "orange"};

    std::unordered_map<std::string, double> idf = calculateIDF(documents);

    std::vector<double> queryVector = calculateTFIDF(query, idf);

    std::vector<std::pair<size_t, double>> similarities; // Store index of document and its similarity

    for (size_t i = 0; i < documents.size(); ++i) {
        std::vector<double> documentVector = calculateTFIDF(documents[i], idf);
        double similarity = cosineSimilarity(queryVector, documentVector);
        similarities.push_back(std::make_pair(i, similarity));
    }

    // Sort similarities in descending order
    std::sort(similarities.begin(), similarities.end(),
              [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                  return a.second > b.second;
              });

    // Output the document closest to the query
    std::cout << "Document closest to the query: " << similarities[0].first << std::endl;

    return 0;
}
