import React, { useState } from 'react';
import { ChakraProvider, Box, Container, Heading, VStack, Button, useToast } from '@chakra-ui/react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [summary, setSummary] = useState(null);
  const toast = useToast();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      toast({
        title: 'Error',
        description: 'Please select a file to upload',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsLoading(true);
      const response = await axios.post('http://localhost:8000/api/v1/summarize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setSummary(response.data);
      toast({
        title: 'Success',
        description: 'Medical record processed successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error uploading file:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to process medical record',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ChakraProvider>
      <Box minH="100vh" bg="gray.50" py={10}>
        <Container maxW="container.lg">
          <VStack spacing={8} align="stretch">
            <Box textAlign="center">
              <Heading as="h1" size="xl" mb={2}>
                AI Medical Record Summarizer
              </Heading>
              <Box as="p" color="gray.600">
                Upload a medical record to generate a summary and extract key information
              </Box>
            </Box>

            <Box bg="white" p={6} rounded="lg" shadow="md">
              <form onSubmit={handleSubmit}>
                <VStack spacing={4}>
                  <Box w="100%">
                    <input
                      type="file"
                      onChange={handleFileChange}
                      accept=".pdf,.txt,.docx"
                      id="file-upload"
                      style={{ display: 'none' }}
                    />
                    <Button
                      as="label"
                      htmlFor="file-upload"
                      colorScheme="blue"
                      variant="outline"
                      cursor="pointer"
                      w="100%"
                      py={8}
                      borderStyle="dashed"
                    >
                      {file ? file.name : 'Choose a file (PDF, TXT, DOCX)'}
                    </Button>
                  </Box>
                  <Button
                    type="submit"
                    colorScheme="blue"
                    isLoading={isLoading}
                    loadingText="Processing..."
                    width="100%"
                    size="lg"
                  >
                    Process Record
                  </Button>
                </VStack>
              </form>
            </Box>

            {summary && (
              <Box bg="white" p={6} rounded="lg" shadow="md">
                <VStack spacing={6} align="stretch">
                  <Box>
                    <Heading as="h2" size="md" mb={4}>
                      Summary
                    </Heading>
                    <Box bg="gray.50" p={4} rounded="md">
                      {summary.summary}
                    </Box>
                  </Box>

                  <Box>
                    <Heading as="h3" size="sm" mb={3}>
                      Key Entities
                    </Heading>
                    <Box display="flex" flexWrap="wrap" gap={2}>
                      {summary.entities?.map((entity, index) => (
                        <Box
                          key={index}
                          bg="blue.100"
                          color="blue.800"
                          px={3}
                          py={1}
                          rounded="full"
                          fontSize="sm"
                        >
                          {entity.text} ({entity.label})
                        </Box>
                      ))}
                    </Box>
                  </Box>
                </VStack>
              </Box>
            )}
          </VStack>
        </Container>
      </Box>
    </ChakraProvider>
  );
}

export default App;
