from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import numpy as np
from gym import spaces

from large_rl.commons.utils import logging


class CandidateSet(object):
    """Class to represent a collection of AbstractDocuments.

       The candidate set is represented as a hashmap (dictionary), with documents
       indexed by their document ID.
    """

    def __init__(self, args: dict):
        """Initializes a document candidate set with 0 documents."""
        self._documents = dict()
        self._cat_doc_dict = dict()
        self._args = args
        self._seed = args["env_seed"]
        self._items_list = list(range(args["num_all_actions"]))

    def size(self):
        """Returns an integer, the number of documents in this candidate set."""
        return len(self._documents)

    def get_all_documents(self):
        """Returns all documents."""
        return self.get_documents(self._documents.keys())

    def get_documents(self, document_ids):
        """Gets the documents associated with the specified document IDs.

        Args:
          document_ids: an array representing indices into the candidate set.
            Indices can be integers or string-encoded integers.

        Returns:
          (documents) an ordered list of AbstractDocuments associated with the
            document ids.
        """
        if type(document_ids) not in [np.ndarray, list]:
            document_ids = [document_ids]
        return [self._documents[int(k)] for k in document_ids]

    def get_categories(self, document_ids):
        """Gets the document categories given doc ids"""
        if type(document_ids) not in [np.ndarray, list]:
            document_ids = [document_ids]
        return [self._documents[int(k)].cluster_id for k in document_ids]

    def add_document(self, document):
        """Adds a document to the candidate set."""
        self._documents[document.doc_id()] = document
        if document.cluster_id not in self._cat_doc_dict:
            self._cat_doc_dict[document.cluster_id] = [document.doc_id()]
        else:
            self._cat_doc_dict[document.cluster_id].append(document.doc_id())

    def remove_document(self, document):
        """Removes a document from the set (to simulate a changing corpus)."""
        del self._documents[document.doc_id()]

    def create_observation(self, if_task_embed: bool = False):
        """Returns a dictionary of observable features of documents."""
        # External API: Used to provide item-embedding for Agent
        return {
            str(k): self._documents[k].create_observation(if_task_embed=if_task_embed) for k in self._documents.keys()
        }

    def create_observation_np(self, if_task_embed: bool = False):
        return np.asarray(list(self.create_observation(if_task_embed=if_task_embed).values()))

    def observation_space(self):
        return spaces.Dict({
            str(k): self._documents[k].observation_space()
            for k in self._documents.keys()
        })

    @property
    def items_list(self):
        return self._items_list


@six.add_metaclass(abc.ABCMeta)
class AbstractDocumentSampler(object):
    """Abstract class to sample documents."""

    def __init__(self, doc_ctor, seed=0, args=None):
        self._doc_ctor = doc_ctor
        self._seed = args["env_seed"]
        self._args = args
        self.reset_sampler()

    @property
    def seed(self):
        return self._seed

    def reset_sampler(self):
        self._rng = np.random.RandomState(self._seed)

    @abc.abstractmethod
    def sample_document(self, **kwargs):
        """Samples and return an instantiation of AbstractDocument."""

    def get_doc_ctor(self):
        """Returns the constructor/class of the documents that will be sampled."""
        return self._doc_ctor

    @property
    def num_clusters(self):
        """Returns the number of document clusters. Returns 0 if not applicable."""
        return 0

    def update_state(self, documents, responses):
        """Update document state (if needed) given user's (or users') responses."""
        pass


@six.add_metaclass(abc.ABCMeta)
class AbstractDocument(object):
    """Abstract class to represent a document and its properties."""

    # Number of features to represent the document.
    NUM_FEATURES = None

    def __init__(self, doc_id):
        self._doc_id = doc_id  # Unique identifier for the document

    def doc_id(self):
        """Returns the document ID."""
        return self._doc_id

    @abc.abstractmethod
    def create_observation(self):
        """Returns observable properties of this document as a float array."""

    @classmethod
    @abc.abstractmethod
    def observation_space(cls):
        """Gym space that defines how documents are represented."""
